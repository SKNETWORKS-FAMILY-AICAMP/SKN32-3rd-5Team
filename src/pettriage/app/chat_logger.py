"""채팅 이력 저장 — `chat_messages` 테이블에 사용자 발화와 시스템 응답을 기록한다.

설계 근거: docs/02 §12 · docs/05 §3 · docs/06 D-36

    **되묻기 슬롯은 저장하지 않는다** (`models.py` 참조 — 이미 세션 스키마에서 제거됨).
    저장하는 것은 **대화 로그** 하나뿐이다 — 평가·오류 분석·과소평가율 추적 (D-13) 근거.

    **DB 미설정이면 조용히 건너뛴다.** 데모 모드에서도 앱이 뜨는 것이 기본 구성이다.
    저장이 실패해도 응답 자체는 나가야 한다 — 로깅이 서비스를 죽이면 안 된다.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sqlalchemy import func
from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from .contracts import AskResponse

log = logging.getLogger(__name__)


def log_chat_turn(
    db: DbSession | None,
    session_id: str,
    user_text: str,
    response: AskResponse,
    pet_id: str | None = None,
) -> None:
    """대화 한 턴(사용자 발화 + 시스템 응답)을 저장한다.

    - DB 세션이 없으면 건너뛴다 (데모 모드).
    - 세션 행이 없으면 생성한다.
    - `seq` 는 기존 최댓값 + 1.
    - **저장 실패는 응답을 막지 않는다** — 로그만 남기고 넘어간다.
    """
    if db is None:
        return

    try:
        from .models import ChatMessage, ChatSession

        # 세션 행이 없으면 만든다.
        session_row = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        if session_row is None:
            session_row = ChatSession(session_id=session_id, pet_id=pet_id)
            db.add(session_row)
            db.flush()

        # 현재 세션의 최대 seq.
        last_seq = (
            db.query(func.max(ChatMessage.seq))
            .filter(ChatMessage.session_id == session_id)
            .scalar()
            or 0
        )

        # 사용자 발화
        db.add(
            ChatMessage(
                session_id=session_id,
                seq=last_seq + 1,
                role="user",
                content=user_text,
            )
        )

        # 시스템 응답
        assistant_text = _extract_response_text(response)
        triage_level = response.triage.level if response.triage else None
        db.add(
            ChatMessage(
                session_id=session_id,
                seq=last_seq + 2,
                role="assistant",
                content=assistant_text,
                response_status=response.status,
                triage_level=triage_level,
            )
        )

        db.commit()
    except SQLAlchemyError as e:
        # 저장 실패는 응답을 막지 않는다.
        log.warning("chat_messages 저장 실패 (SQL) — session=%s: %s", session_id, type(e).__name__)
        try:
            db.rollback()
        except Exception:
            pass
    except Exception as e:  # noqa: BLE001 — 어떤 이유든 응답을 막지 않는다.
        log.warning("chat_messages 저장 실패 — session=%s: %s", session_id, type(e).__name__)


def _extract_response_text(response: AskResponse) -> str:
    """AskResponse 에서 어떤 상태든 사용자에게 노출되는 문장을 뽑는다."""
    if response.answer:
        return response.answer
    if response.clarify:
        return response.clarify.question
    if response.refusal:
        return f"{response.refusal.message} {response.refusal.advice}"
    return ""