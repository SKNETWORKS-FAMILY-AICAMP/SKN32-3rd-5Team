"""POST /api/ask — 질의응답 (02 §12 질의응답 화면).

응답은 항상 HTTP 200이다. `status` 가 answered / clarify / refused 를 나른다.

    거절을 4xx로 내보내지 않는 이유: 거절은 오류가 아니라 **설계된 정상 경로**다
    (02 §9). 4xx로 만들면 프론트가 에러 핸들러에서 처리하게 되고,
    거절 화면이 "장애 화면"처럼 보이게 된다.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ..contracts import AskRequest, AskResponse
from ..deps import get_engine, get_sessions
from ..engine import QAEngine, refuse
from ..session import SessionStore

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    engine: QAEngine = Depends(get_engine),
    sessions: SessionStore = Depends(get_sessions),
) -> AskResponse:
    session = sessions.get_or_create(req.session_id)
    try:
        return engine.ask(req, session)
    except Exception as e:
        # 엔진이 예외로 죽어도 단정적인 답을 흘리지 않는다.
        #
        # ⚠️ 예외 **메시지**를 로그에 넣지 않는다 — LLM·DB 클라이언트는 프롬프트를
        #    예외에 담는 일이 흔해서, 그대로 찍으면 질문 원문이 로그로 샌다 (D-36).
        #    디버깅에는 예외 타입과 session_id 면 시작점으로 충분하고,
        #    본문이 필요하면 개발 환경에서 트레이스를 따로 켠다.
        log.error(
            "engine failure — type=%s session=%s (메시지는 개인정보 우려로 생략)",
            type(e).__name__,
            session.session_id,
        )
        return refuse(
            session,
            "판정불가",
            "요청을 처리하지 못했습니다. 증상이 의심되면 지체하지 마세요.",
        )
