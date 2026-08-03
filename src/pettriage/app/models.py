"""SQLAlchemy ORM 모델 — users / pets / chat_sessions / chat_messages.

설계 근거: docs/06 D-36 · docs/05 §3

**되묻기 슬롯은 여기 들어오지 않는다.**

    `ChatSession` 에 `clarify_turns`·`weight_kg`·`amount_g` 가 있었다.
    그런데 `app/session.py` 는 같은 값을 두고 이렇게 적어 두었다 —

    > 프로세스가 죽으면 사라지는 것이 **의도**다.
    > 되묻기 슬롯(체중·섭취량)은 **보관할 이유가 없다** (D-36 최소 수집).

    같은 값을 한쪽은 휘발이 의도라 하고 다른 쪽은 테이블에 넣고 있었다.
    체중·섭취량 시계열은 D-36 표가 *"누적되면 생활 패턴이 드러난다"* 로
    지목한 바로 그 항목이다. 세 칼럼을 뺐다 (2026-08-01 PR#3 검수).

    **대화 로그(`ChatMessage`)는 남긴다** — 평가·오류 분석과
    D-13 과소평가율 추적의 근거이고, 슬롯과 달리 목적이 분명하다.

시각은 **timezone-aware UTC** 로 저장하고 표시 계층에서 KST 로 바꾼다.
`datetime.utcnow()` 는 tz 정보가 없어, 변환을 한 번만 잊어도 **9시간 틀린다** —
06 문서 전체가 UTC로 찍혀 하루씩 앞당겨졌던 것과 같은 종류의 사고다.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    """tz 를 붙여 둔다. naive 로 저장하면 KST 변환을 잊는 순간 9시간 틀린다."""
    return datetime.now(UTC)


class User(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str] = mapped_column(String(50), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pets: Mapped[list[Pet]] = relationship("Pet", back_populates="user")


class Pet(Base):
    __tablename__ = "pets"

    pet_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.user_id"), nullable=False)
    name: Mapped[str] = mapped_column(String(50), nullable=False)  # 반려동물 이름
    species: Mapped[str] = mapped_column(String(10), nullable=False)  # dog / cat / bird
    breed: Mapped[str | None] = mapped_column(String(50), nullable=True)  # 품종
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship("User", back_populates="pets")
    chat_sessions: Mapped[list[ChatSession]] = relationship("ChatSession", back_populates="pet")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    session_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    pet_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("pets.pet_id", ondelete="CASCADE"), nullable=True
    )
    # ⚠ clarify_turns · weight_kg · amount_g 를 **의도적으로 두지 않는다.**
    #   되묻기 슬롯은 휘발이 의도다 (D-36 최소 수집 · 05 §3). session.py 가 메모리에 들고 있다.
    #   여기 되살리려면 D-36 을 뒤집는 결정이므로 06 에 기록부터 해야 한다 (D-22).
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    pet: Mapped[Pet | None] = relationship("Pet", back_populates="chat_sessions")
    messages: Mapped[list[ChatMessage]] = relationship("ChatMessage", back_populates="session")


class DiaryEntry(Base):
    """일일 다이어리 기록.

    설계 근거: docs/02 §5 · §12 · docs/00 §3.3 (킬러 기능 - 기간 리포트)

    **일반 웹 앱 기능이다** (00 §3.3) — 벡터DB 적재는 별도 (WS1 담당).
    여기는 화면 조회·기간 리포트용 원본이다.

    소유자 확인: `(user_id, pet_id)` 조건이 모든 조회에 붙어야 한다 —
    `pet_id` 만으로 찾으면 남의 기록을 읽는다 (D-52 원칙).

    `weight_kg` 는 nullable — 매일 재지 않는다. NULL 을 그대로 둔다 (D-52).
    """

    __tablename__ = "diary_entries"

    entry_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    pet_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("pets.pet_id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[str] = mapped_column(String(36), ForeignKey("users.user_id"), nullable=False)
    recorded_date: Mapped[date] = mapped_column(Date, nullable=False)  # 캘린더용 (날짜만)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    meals: Mapped[str | None] = mapped_column(Text, nullable=True)      # JSON array
    symptoms: Mapped[str | None] = mapped_column(Text, nullable=True)   # JSON array
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    droppings: Mapped[str | None] = mapped_column(String(100), nullable=True)  # 조류 전용
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"
    #: 같은 세션에 같은 순번이 둘이면 대화 순서가 뒤집힌다.
    __table_args__ = (UniqueConstraint("session_id", "seq", name="uq_message_seq"),)

    message_id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid4())
    )
    session_id: Mapped[str] = mapped_column(
        String(32), ForeignKey("chat_sessions.session_id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(10), nullable=False)  # user / assistant
    content: Mapped[str] = mapped_column(Text, nullable=False)
    response_status: Mapped[str | None] = mapped_column(
        String(20), nullable=True
    )  # answered / clarify / refused
    triage_level: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1~4
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[ChatSession] = relationship("ChatSession", back_populates="messages")
