from http.client import responses

from Generative_AI import *




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
    print(user_input)

    Human_message=HumanMessagePromptTemplate.from_template(
        "Help fulfilling the user's need for the following task here:{task}",input_variables=["task"]
    )
    prompt=ChatPromptTemplate.from_messages([system_prompt,Human_message])
    print(prompt.format(task=user_input))
    chain_one = (
            {
                "task": lambda x: x["task"]
            }
            | prompt
            | model
            | {"response": lambda x: x.content}
    )
    res = chain_one.invoke({
        "task": user_input

    })






    return f"🤖 AI processed: {res['response']}"