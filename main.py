




from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from fastapi import FastAPI, Depends, HTTPException, Request, Response, Form
from starlette.responses import HTMLResponse


#Setting Up Templates
templates = Jinja2Templates(directory="templates")
app=FastAPI()

# Global chat memory (in-memory for demo)
messages = []

@app.get("/", response_class=HTMLResponse)
async def get_chat(request: Request):
    # If it's the first visit, add the welcome message
    if not messages:
        messages.append({"role": "assistant", "content": "Hey, I am your AI agent. How can I help you today?"})

    return templates.TemplateResponse("chat.html", {"request": request, "messages": messages})

@app.post("/", response_class=HTMLResponse)
async def process_message(request: Request, message: str = Form(...)):
    # Store user message
    messages.append({"role": "user", "content": message})


    reply = await run_agent(message)

    # Store assistant reply
    messages.append({"role": "assistant", "content": reply})

    return templates.TemplateResponse("chat.html", {"request": request, "messages": messages})



async def run_agent(user_input: str) -> str:
    return f"🤖 AI processed: {user_input}"