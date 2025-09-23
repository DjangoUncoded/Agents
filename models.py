from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class Chat(Base):
    __tablename__ = 'chats'

    id = Column(Integer, primary_key=True, autoincrement=True)
    summary = Column(Text, nullable=True,default='This is my very first time talking to you')
    created_at = Column(DateTime, default=datetime.utcnow)
    username = Column(String(50), nullable=False, unique=True, index=True)



    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    # Relationship back to User
    user = relationship("User", back_populates="chats")


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    password = Column(String(255), nullable=False)  # store hashed passwords
    created_at = Column(DateTime, default=datetime.utcnow)

    # A user can have many chats
    chats = relationship(
        "Chat",
        back_populates="user",
        cascade="all, delete-orphan"
    )
