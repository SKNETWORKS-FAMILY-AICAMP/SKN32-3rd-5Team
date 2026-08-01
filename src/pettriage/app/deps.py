"""의존성 주입 지점.

엔진 교체는 **이 파일 한 곳**이다 (D-40).

    WS2의 LangGraph 그래프가 완성되면 `_build_engine` 의 `graph` 분기가
    `GraphEngine` 을 찾아 쓴다. 라우터·계약·프론트·테스트는 그대로다.
    테스트는 `app.dependency_overrides[get_engine]` 로 임의 엔진을 끼운다.

어느 엔진을 쓸지는 코드가 아니라 **설정**이 정한다 — `configs/*.yaml` 의 `serve.engine`.
"""

from __future__ import annotations

import logging
import os

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..config import get_config
from .engine import QAEngine, StubEngine
from .records_store import RecordStore
from .session import SessionStore

log = logging.getLogger(__name__)

_engine: QAEngine | None = None
_sessions = SessionStore()
_records = RecordStore()


class EngineUnavailable(RuntimeError):
    """설정이 요구한 엔진을 만들 수 없다.

    조용히 스텁으로 내려가면 **평가 지표가 스텁으로 산출된다.**
    그 지표는 오염된 것이므로 기본은 실패다.
    시연 중 급하면 `PETTRIAGE_ALLOW_ENGINE_FALLBACK=1` 로 낮출 수 있다.
    """


def _build_engine() -> QAEngine:
    kind = get_config().serve.engine
    if kind == "graph":
        try:
            from ..graph.engine import EngineNotReady, GraphEngine

            return GraphEngine()
        except (ImportError, EngineNotReady) as e:
            msg = (
                "serve.engine=graph 인데 GraphEngine 을 쓸 수 없다 "
                f"({type(e).__name__}). 스텁으로 기동하면 평가 결과가 오염된다 (04 §8)."
            )
            if os.getenv("PETTRIAGE_ALLOW_ENGINE_FALLBACK") != "1":
                raise EngineUnavailable(msg) from e
            log.warning("%s PETTRIAGE_ALLOW_ENGINE_FALLBACK=1 이라 스텁으로 진행한다.", msg)
    return StubEngine()


def get_engine() -> QAEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_sessions() -> SessionStore:
    return _sessions


def get_records() -> RecordStore:
    return _records


def set_engine(engine: QAEngine | None) -> None:
    """부팅 시점 교체용. `None` 을 넣으면 다음 호출에서 다시 만든다 (테스트용)."""
    global _engine
    _engine = engine


def reset_state() -> None:
    """프로세스 전역 상태를 비운다. **테스트 전용.**

    엔진·세션·기록이 모듈 전역이라 테스트가 서로를 오염시킬 수 있다.
    """
    global _engine
    _engine = None
    _sessions.clear()
    _records.clear()


def allowed_origins() -> list[str]:
    """CORS 허용 출처.

    기본 구성은 FastAPI가 프론트를 같은 출처에서 서빙하므로 **CORS가 필요 없다.**
    별도 개발 서버를 띄울 때만 `configs/*.yaml` 의 `serve.cors_origins` 에 나열한다.
    와일드카드를 기본값으로 두지 않는다.
    """
    return [o for o in get_config().serve.cors_origins if o and o != "*"]


# ─────────────────────────────────────────────────────────────
# DB · 인증 주입 (WS5 백엔드)
#
# 라우터가 `database.get_db` 를 직접 부르지 않는다 — **주입 지점은 이 파일 하나다** (D-40).
# 테스트가 `app.dependency_overrides[get_db]` 로 인메모리 세션을 끼울 수 있어야 하고,
# 그러려면 라우터가 참조하는 심볼이 한 곳이어야 한다.
# ─────────────────────────────────────────────────────────────
def get_db():
    """DB 세션. `database.get_db` 를 감싼다.

    여기서 임포트하는 이유 — 모듈 최상단에서 하면 `[db]` extra 없이는
    `deps` 전체가 임포트 실패한다. DB 를 안 쓰는 구성이 정상이므로 그러면 안 된다.
    """
    from .database import get_db as _get_db

    yield from _get_db()


#: 모듈 전역 싱글턴. 인자 기본값에서 호출하면 요청마다 새로 만들어진다 (B008).
_bearer = HTTPBearer()
_bearer_dep = Depends(_bearer)


def get_current_user_id(credentials: HTTPAuthorizationCredentials = _bearer_dep) -> str:
    """Bearer 토큰 → `user_id`. 실패하면 401.

    **`jwt` 를 여기서 임포트하지 않는다** (D-40). `app.auth` 가 우리 예외로 번역해 주고
    이 함수는 그것만 본다. 라이브러리를 바꿔도 배달 계층은 그대로다.

    실패 사유를 나눠 말하되 **원본 예외는 `from None` 으로 끊는다** —
    스택에 서명 키·알고리즘이 실려 응답이나 로그로 새면 안 된다.
    """
    from .auth import TokenExpiredError, TokenInvalidError, decode_access_token

    try:
        return decode_access_token(credentials.credentials)
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="토큰이 만료되었습니다."
        ) from None
    except TokenInvalidError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="유효하지 않은 토큰입니다."
        ) from None
