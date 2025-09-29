
import asyncio

from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from models import *
load_dotenv()
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)

# Load API key
API_KEY = os.getenv("GEMINI_API")

# Set up model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

# Define system message template
system_prompt = "You are an AI agent, designed and implemented for Youth Mental Wellness Task."

# Adding Chat Memoization
from langchain.memory import ConversationSummaryMemory
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

class ConversationSummaryMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: "ChatGoogleGenerativeAI"
    _username: str
    _user_id: int  # actual logged-in user ID

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, llm: "ChatGoogleGenerativeAI", username: str, user_id: int, summary: str = ""):
        super().__init__(llm=llm)
        self._username = username
        self._user_id = user_id
        if summary:
            self.messages = [SystemMessage(content=summary)]

    async def add_messages(self, messages: list[BaseMessage]):
        """
        Add messages and update conversation summary in memory and DB.
        Creates its own DB session to avoid conflicts.
        """
        existing_summary = self.messages[0].content if self.messages else ""

        # Build summary prompt
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensure to maintain "
                "as much relevant information as possible."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{messages}"
            )
        ])

        formatted_messages = summary_prompt.format_messages(
            existing_summary=existing_summary,
            messages=[x.content for x in messages]
        )

        try:
            if hasattr(self.llm, "ainvoke"):
                new_summary = await self.llm.ainvoke(formatted_messages)
            else:
                new_summary = self.llm.invoke(formatted_messages)
        except Exception as e:
            print(f"Error during LLM invocation: {e}")
            new_summary_content = f"{existing_summary}\n\n{'; '.join([x.content for x in messages])}"
            new_summary = type('MockResponse', (), {'content': new_summary_content})()

        self.messages = [SystemMessage(content=new_summary.content)]

        # Persist to DB - import here to avoid circular imports
        try:
            from database import SessionLocal
            async with SessionLocal() as db:
                stmt = select(Chat).where(Chat.username == self._username)
                result = await db.execute(stmt)
                chat = result.scalar_one_or_none()

                if chat:
                    chat.summary = new_summary.content
                else:
                    chat = Chat(username=self._username, summary=new_summary.content, user_id=self._user_id)
                    db.add(chat)

                await db.commit()
        except Exception as e:
            print(f"Error during database operation: {e}")
            import traceback
            traceback.print_exc()
    def clear(self) -> None:
        self.messages = []