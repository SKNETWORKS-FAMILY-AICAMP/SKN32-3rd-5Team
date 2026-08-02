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
from .safety_engine import SafetyEngine
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
    """설정이 가리키는 엔진을 만든다.

    ⚠️ `except` 절에 `EngineNotReady` 를 쓰지 않는다.

        try:
            from ..graph.engine import EngineNotReady, GraphEngine
            return GraphEngine()
        except (ImportError, EngineNotReady) as e:      # ← 이렇게 짜여 있었다

    `EngineNotReady` 는 **같은 `try` 가 바인딩하는 지역 이름**이다. 그 임포트가
    `ImportError` 를 내면 except 튜플을 평가하는 시점에 이름이 없어서
    `UnboundLocalError` 가 난다 — **크게 실패하지도(`EngineUnavailable`),
    폴백하지도 못한다.** 2026-08-02 검토에서 재현했다.

    넓은 `except Exception` 이 여기서는 옳다. **graph 를 못 만든 이유가 무엇이든
    결과는 둘 중 하나**이기 때문이다 — 크게 실패하거나(기본), 명시적으로 폴백하거나.
    """
    kind = get_config().serve.engine
    if kind == "graph":
        try:
            from ..graph.engine import GraphEngine

            return GraphEngine()
        except Exception as e:  # noqa: BLE001 — ImportError · EngineNotReady · 그 밖의 기동 실패
            msg = (
                "serve.engine=graph 인데 GraphEngine 을 쓸 수 없다 "
                f"({type(e).__name__}). 스텁으로 기동하면 평가 결과가 오염된다 (04 §8)."
            )
            if os.getenv("PETTRIAGE_ALLOW_ENGINE_FALLBACK") != "1":
                raise EngineUnavailable(msg) from e
            log.warning("%s PETTRIAGE_ALLOW_ENGINE_FALLBACK=1 이라 스텁으로 진행한다.", msg)
    return StubEngine()


def get_engine() -> QAEngine:
    """엔진을 만들고 **반드시 `SafetyEngine` 으로 감싼다** (D-47).

    감싸는 일을 여기서 하는 이유 — 엔진 구현체가 각자 `scrub_contacts` 를 부르게 두면
    **새 엔진을 꽂는 사람이 모르면 그대로 뚫린다.** 실제로 그렇게 뚫려 있었다
    (2026-08-02 검토: 유일한 호출부가 만들어지지도 않는 `GraphEngine` 안에 있었다).

    D-40 — *지키기로 한 것이 아니라 못 어기는 것.* 주입 지점이 그 자리다.
    """
    global _engine
    if _engine is None:
        _engine = SafetyEngine(_build_engine())
    return _engine


def get_sessions() -> SessionStore:
    return _sessions


def get_records() -> RecordStore:
    return _records


def set_engine(engine: QAEngine | None) -> None:
    """부팅 시점 교체용. `None` 을 넣으면 다음 호출에서 다시 만든다 (테스트용).

    넣은 엔진도 **`SafetyEngine` 으로 감싼다.** 테스트가 가짜 엔진을 꽂았다고 해서
    D-47 이 꺼지면, *"연락처가 나가는지"* 를 검증하는 테스트 자체가 성립하지 않는다.
    이미 감싼 것을 또 감싸지는 않는다.
    """
    global _engine
    _engine = engine if engine is None or isinstance(engine, SafetyEngine) else SafetyEngine(engine)


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
#:
#: ⚠️ **`auto_error=False` 다. 상태 코드를 라이브러리에 맡기지 않는다.**
#:
#: 기본값(`auto_error=True`)이면 토큰이 없을 때 FastAPI 가 직접 예외를 낸다.
#: 그런데 그 코드가 버전마다 다르다 — `0.115.6`(핀된 값)은 **403**,
#: `0.13x` 이상은 **401**. 2026-08-02 에 이것 때문에
#: **로컬은 초록인데 CI 는 빨간** 상태가 만들어졌다.
#:
#: 값이 갈리는 것보다 나쁜 것은 **어느 쪽이 우리 결정인지 아무도 모르는 것**이다.
#: 우리는 401 로 정한다 — 403 은 *"누군지 알지만 권한이 없다"* 이고,
#: 여기는 **누군지조차 모른다.** 그 결정을 우리 코드에 적는다 (D-40).
_bearer = HTTPBearer(auto_error=False)
_bearer_dep = Depends(_bearer)


def get_current_user_id(credentials: HTTPAuthorizationCredentials | None = _bearer_dep) -> str:
    """Bearer 토큰 → `user_id`. 실패하면 **401** (인증 정보 없음·만료·위조 모두).

    **`jwt` 를 여기서 임포트하지 않는다** (D-40). `app.auth` 가 우리 예외로 번역해 주고
    이 함수는 그것만 본다. 라이브러리를 바꿔도 배달 계층은 그대로다.

    실패 사유를 나눠 말하되 **원본 예외는 `from None` 으로 끊는다** —
    스택에 서명 키·알고리즘이 실려 응답이나 로그로 새면 안 된다.
    """
    from .auth import TokenExpiredError, TokenInvalidError, decode_access_token

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="인증이 필요합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
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


#: DB 없는 데모 구성의 단일 소유자.
#: 그 구성에는 **사용자 개념 자체가 없으므로** 넘나들 상대도 없다.
DEMO_OWNER = "demo"


def get_owner_id(creds: HTTPAuthorizationCredentials | None = _bearer_dep) -> str:
    """다이어리 기록의 **소유자**. 없으면 401 (DB 구성에 한해).

    두 구성을 구분한다 — `routes/__init__.py` 가 라우터를 고르는 규칙과 같다.

    ==================  ====================================================
    `DATABASE_URL`      동작
    ==================  ====================================================
    있음                Bearer 토큰 필수. 토큰의 `sub` 가 소유자다.
                        **없으면 401** — 남의 다이어리를 열 수 없다
    없음                `DEMO_OWNER` 단일 소유자. 사용자 개념이 없는 구성이다
    ==================  ====================================================

    예전에는 `records` 라우터에 인증 의존성이 아예 없었다. `pets` 는 모든 쿼리에
    `user_id` 를 붙이는데 여기만 빠져서, **`pet_id` 만 알면 남의 기록을 읽었다**
    (2026-08-02 재현). `records_store.py` 주석에 *"인증이 없다"* 고 적혀 있던 것이
    오히려 *"고쳐도 되는 것"* 을 가렸다.
    """
    if not os.getenv("DATABASE_URL"):
        return DEMO_OWNER
    if creds is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="다이어리는 로그인이 필요합니다.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 토큰 해석은 `get_current_user_id` 한 곳에서만 한다 (D-40).
    return get_current_user_id(creds)
