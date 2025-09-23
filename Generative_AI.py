from dotenv import load_dotenv
load_dotenv()
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,MessagesPlaceholder
)

# Load API key
API_KEY = os.getenv("GEMINI_API")

# Set up model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=API_KEY)

# Define system message template
system_prompt =  "You are an AI agent,designed and implemented for Youth Mental Wellness Task."


#Adding Chat Memoization

from langchain.memory import (ConversationSummaryMemory)
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec



class ConversationSummaryMessageHistory(BaseChatMessageHistory, BaseModel):
    messages: list[BaseMessage] = Field(default_factory=list)
    llm: "ChatGoogleGenerativeAI"  # your LLM type here

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, llm: "ChatGoogleGenerativeAI", db: AsyncSession, username: str, summary: str = ""):
        super().__init__(llm=llm)
        self._db = db            # private attribute, not part of BaseModel fields
        self._username = username
        if summary:
            self.messages = [SystemMessage(content=summary)]

    async def add_messages(self, messages: list[BaseMessage]) -> None:
        """Add messages and update conversation summary in memory and DB."""
        # Current summary
        existing_summary = self.messages[0].content if self.messages else ""

        # Construct summary prompt
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

        # Invoke LLM asynchronously to get new summary
        new_summary = await self.llm.ainvoke(
            summary_prompt.format_messages(
                existing_summary=existing_summary,
                messages=[x.content for x in messages]
            )
        )

        # Update in-memory summary
        self.messages = [SystemMessage(content=new_summary.content)]

        # Persist summary in DB
        stmt = select(Chat).where(Chat.username == self._username)
        result = await self._db.execute(stmt)
        chat = result.scalar_one_or_none()

        if chat:
            chat.summary = new_summary.content
        else:
            # If no chat exists, create new
            chat = Chat(username=self._username, summary=new_summary.content, user_id=1)  # <-- set proper user_id
            self._db.add(chat)

        await self._db.commit()

    def clear(self) -> None:
        """Clear the memory messages."""
        self.messages = []

