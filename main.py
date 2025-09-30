import asyncio
import os
from datetime import datetime, timedelta
from typing import Optional
from functools import partial

from fastapi import FastAPI, Depends, HTTPException, Request, Response, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.responses import HTMLResponse
import traceback

from pydantic import BaseModel
from jose import jwt, JWTError
from passlib.context import CryptContext

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session
from sqlalchemy.sql.expression import text, select

# --- Local Imports ---
import models
from models import *
from database import SessionLocal, engine
from Generative_AI import *
from langchain_core.messages import HumanMessage, AIMessage

if os.name == 'nt':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# --- App Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Global chat memory (in-memory for demo)
messages = []
chat_map = {}

# --- Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("❌ Exception:", exc)
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )

# --- Database Dependency ---
async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# --- Authentication ---
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str):
    return bcrypt_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def authenticate_user(username: str, password: str, db: AsyncSession = Depends(get_db)):
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user or not verify_password(password, user.password):
        return None
    return user

# ------------------- MODIFICATION START -------------------
# get_current_user now uses the dependency-injected 'db' session.
async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)):
    """Gets current user using the request's DB session."""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: No username")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token: JWTError")

    # It no longer creates its own session.
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
# -------------------- MODIFICATION END --------------------

# --- Routes ---

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("Login.html", {"request": request})

@app.post("/")
async def login(response: Response, username: str = Form(...), password: str = Form(...), db: AsyncSession = Depends(get_db)):
    user = await authenticate_user(username, password, db)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})
    response = RedirectResponse(url="/protected", status_code=303)
    response.set_cookie(
        key="access_token", value=access_token, httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60, secure=False, samesite="lax", path="/"
    )
    return response

@app.get("/signup")
async def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
async def signup(username: str = Form(...), password: str = Form(...), email: str = Form(...), db: AsyncSession = Depends(get_db)):
    user = await db.execute(select(User).where(User.username == username))
    if user.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Username already exists")

    existing_email = await db.execute(select(User).where(User.email == email))
    if existing_email.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    created_user = User(username=username, password=bcrypt_context.hash(password), email=email)
    created_chat = Chat(username=username, user=created_user)
    db.add(created_user)
    db.add(created_chat)
    await db.commit()
    return RedirectResponse(url="/", status_code=303)

@app.get("/protected", response_class=HTMLResponse)
async def get_chat(request: Request, user: User = Depends(get_current_user)):
    if not messages:
        messages.append({"role": "assistant", "content": "Hey, I am your AI agent. How can I help you today?"})
    return templates.TemplateResponse("chat.html", {"request": request, "messages": messages})

# ------------------- MODIFICATION START -------------------
# The endpoint now depends on 'get_current_user' to get the user object.
@app.post("/protected", response_class=HTMLResponse)
async def process_message(
    request: Request,
    message: str = Form(...),
    db: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user)  # Inject user directly
):
    """ Main chat endpoint. """
    ai_reply = await run_agent(message, request, db, user) # Pass user to run_agent

    messages.append({"role": "user", "content": message})
    messages.append({"role": "assistant", "content": ai_reply})

    return templates.TemplateResponse("chat.html", {"request": request, "messages": messages})

async def get_chat_history(session_id: str, llm: "ChatGoogleGenerativeAI", username: str, user_id: int, summary: str):
    if session_id not in chat_map:
        chat_map[session_id] = ConversationSummaryMessageHistory(
            llm=llm, username=username, user_id=user_id, summary=summary
        )
    return chat_map[session_id]

# run_agent now accepts the 'current_user' object directly.
async def run_agent(user_input: str, request: Request, db: AsyncSession, current_user: User) -> str:
    """ Processes user input, manages memory, invokes LLM, and updates DB. """
    try:
        # User is now passed directly into the function
        username = current_user.username
        user_id = current_user.id
        session_id = f"session_{username}"

        stmt = select(Chat).where(Chat.username == username)
        result = await db.execute(stmt)
        chat = result.scalar_one_or_none()
        summary = chat.summary if chat else "This is my very first time talking to you"

        history = await get_chat_history(
            session_id=session_id, llm=model, username=username, user_id=user_id, summary=summary
        )

        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_prompt),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{query}")
        ])

        formatted_messages = prompt_template.format_messages(
            query=user_input, history=history.messages
        )

        if hasattr(model, "ainvoke"):
            res = await model.ainvoke(formatted_messages)
        else:
            res = model.invoke(formatted_messages)

        ai_response = res.content if hasattr(res, "content") else str(res)

        # Pass the 'db' session to the updated add_messages method
        await history.add_messages(
            messages=[
                HumanMessage(content=user_input),
                AIMessage(content=ai_response)
            ],
            db=db
        )

        return ai_response

    except Exception as e:
        print(f"Error in run_agent: {e}")
        traceback.print_exc()
        return "I'm sorry, but I encountered an error while processing your request. Please try again."