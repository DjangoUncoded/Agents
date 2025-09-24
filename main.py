

from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel

from fastapi import FastAPI, Depends, HTTPException, Request, Response, Form
from starlette.responses import HTMLResponse

import  os
from datetime import datetime, timedelta


async def get_db():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()






from Generative_AI import *

#Database
from sqlalchemy.sql.expression import text, select
import models
from models import *
from database import *
from sqlalchemy.orm import Session

from contextlib import asynccontextmanager

@asynccontextmanager
async def get_db_session():
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

#Authentication
from typing import Optional
from jose import jwt, JWTError
from passlib.context import CryptContext
bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

def verify_password(plain_password: str, hashed_password: str):
    return bcrypt_context.verify(plain_password, hashed_password)


async def authenticate_user(username: str, password: str, db:AsyncSession=Depends(get_db)):
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()


    if not user or not verify_password(password, user.password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(request: Request, db: AsyncSession = Depends(get_db)):
    token = request.cookies.get("access_token")
    if not token:
        # This will trigger a redirect to the login page from the frontend if needed
        raise HTTPException(status_code=401, detail="Not authenticated")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: No username")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token: JWTError")

    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
















#Setting Up Templates
templates = Jinja2Templates(directory="templates")
app=FastAPI()

# Global chat memory (in-memory for demo)
messages = []
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("Login.html", {"request": request})

#Debugging

from fastapi.responses import JSONResponse
import traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print("❌ Exception:", exc)
    traceback.print_exc()  # full stack trace in Vercel logs
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)}
    )



@app.post("/")
async def login(
        response: Response,
        username: str = Form(...),
        password: str = Form(...),
        db: Session = Depends(get_db)
):
    user = await authenticate_user(username, password, db)
    if not user:
        # Ideally, you'd return an error message to the login page
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})

    response = RedirectResponse(url="/protected", status_code=303)
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        secure=False,  # Set to True in production with HTTPS
        samesite="lax",
        path="/"
    )
    return response



@app.get("/signup")
async def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@app.post("/signup")
async def signup(
        username: str = Form(...),
        password: str = Form(...),
        email: str = Form(...),

        db: AsyncSession = Depends(get_db),
):
    stmt = select(User).where(User.username == username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user:
        # return an error message to the signup page
        raise HTTPException(status_code=400, detail="Username already exists")

    created_user = User(
        username=username,
        password=bcrypt_context.hash(password),


        email=email,
    )
    created_chat = Chat(
        username=username,
        user=created_user

    )
    db.add(created_chat)
    db.add(created_user)
    await db.commit()
    await db.refresh(created_user)
    await db.refresh(created_chat)
    return RedirectResponse(url="/", status_code=303)














from functools import partial


@app.get("/protected", response_class=HTMLResponse)
async def get_chat(request: Request):
    # If it's the first visit, add the welcome message
    if not messages:
        messages.append({"role": "assistant", "content": "Hey, I am your AI agent. How can I help you today?"})

    return templates.TemplateResponse("chat.html", {"request": request, "messages": messages})

@app.post("/protected", response_class=HTMLResponse)
async def process_message(request: Request, message: str = Form(...), db: AsyncSession = Depends(get_db)):
    reply = await run_agent(message, request, db, db_factory=get_db_session)
    messages.append({"role": "user", "content": message})
    messages.append({"role": "assistant", "content": reply})
    return templates.TemplateResponse("chat.html", {"request": request, "messages": messages})


chat_map = {}

async def get_chat_history(session_id: str, llm: "ChatGoogleGenerativeAI", username: str, user_id: int, summary: str):
    if session_id not in chat_map:
        chat_map[session_id] = ConversationSummaryMessageHistory(
            llm=llm,
            username=username,
            user_id=user_id,
            summary=summary
        )
    return chat_map[session_id]


#Adding Memory Feature
async def run_agent(user_input: str, request: Request, db: AsyncSession, db_factory: callable) -> str:
    # 1️⃣ Get logged-in user
    user = await get_current_user(request, db)
    username = user.username
    user_id = user.id
    session_id = f"session_{username}"

    # 2️⃣ Fetch existing summary safely
    async with db_factory() as tmp_db:
        stmt = select(Chat).where(Chat.username == username)
        result = await tmp_db.execute(stmt)
        chat = result.scalar_one_or_none()
        summary = chat.summary if chat else "This is my very first time talking to you"

    # 3️⃣ Get or create memory
    history = await get_chat_history(
        session_id=session_id,
        llm=model,
        username=username,
        user_id=user_id,
        summary=summary
    )

    # 4️⃣ Build prompt and LLM pipeline
    prompt_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{query}")
    ])
    pipeline = prompt_template | model

    # 5️⃣ Invoke LLM safely in a threadpool
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        executor,
        lambda: pipeline.invoke({
            "query": user_input,
            "history": history.messages
        })
    )
    ai_response = res.content if hasattr(res, "content") else str(res)

    # 6️⃣ Update memory & DB
    await history.add_messages(db_factory, [
        HumanMessage(content=user_input),
        AIMessage(content=ai_response)
    ])

    return ai_response