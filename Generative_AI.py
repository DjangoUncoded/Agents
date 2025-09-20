from dotenv import load_dotenv
load_dotenv()
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Load API key
API_KEY = os.getenv("GEMINI_API")

# Set up model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=API_KEY)

# Define system message template
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an AI agent,designed and implemented for Youth Mental Wellness Task."
)