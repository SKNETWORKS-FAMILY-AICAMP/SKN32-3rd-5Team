"""SQLAlchemy ORM 모델 — users / pets / chat_sessions / chat_messages."""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    pets: Mapped[list[Pet]] = relationship("Pet", back_populates="user")


class Pet(Base):
    __tablename__ = "pets"

    pet_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.user_id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(50), nullable=False)       # 반려동물 이름
    species: Mapped[str] = mapped_column(String(10), nullable=False)    # dog / cat / bird
    breed: Mapped[str | None] = mapped_column(String(50), nullable=True)  # 품종
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped[User] = relationship("User", back_populates="pets")
    chat_sessions: Mapped[list[ChatSession]] = relationship(
        "ChatSession", back_populates="pet"
    )


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    pet_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("pets.pet_id"), nullable=True
    )
    clarify_turns: Mapped[int] = mapped_column(Integer, default=0)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    pet: Mapped[Pet | None] = relationship("Pet", back_populates="chat_sessions")
    messages: Mapped[list[ChatMessage]] = relationship(
        "ChatMessage", back_populates="session"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    message_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("chat_sessions.session_id"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    response_status: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # answered / clarify / refused
    triage_level: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1~4
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")