"""의존성 주입 지점.

엔진 교체는 **이 파일 한 곳**이다 (D-40).

    WS2의 LangGraph 그래프가 완성되면 `_build_engine` 의 `graph` 분기에
    `GraphEngine` 을 연결한다. 라우터·계약·프론트·테스트는 그대로다.
    테스트는 `app.dependency_overrides[get_engine]` 로 임의 엔진을 끼운다.

어느 엔진을 쓸지는 코드가 아니라 **설정**이 정한다 — `configs/*.yaml` 의 `serve.engine`.
"""

from __future__ import annotations

import logging

from ..config import get_config
from .engine import QAEngine, StubEngine
from .session import SessionStore

log = logging.getLogger(__name__)

_engine: QAEngine | None = None
_sessions = SessionStore()


def _build_engine() -> QAEngine:
    kind = get_config().serve.engine
    if kind == "graph":
        try:
            from ..graph.engine import GraphEngine  # type: ignore[attr-defined]

            return GraphEngine()
        except ImportError:
            # 안전한 실패: 그래프가 아직 없으면 스텁으로 내려간다.
            # 조용히 넘어가지 않고 반드시 경고를 남긴다 —
            # 평가 결과가 스텁으로 산출되면 그 자체가 오염이다 (04 §8).
            log.warning(
                "serve.engine=graph 인데 GraphEngine 을 임포트하지 못했다. "
                "스텁으로 기동한다 — 평가 실행이라면 즉시 중단할 것."
            )
    return StubEngine()


def get_engine() -> QAEngine:
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


def get_sessions() -> SessionStore:
    return _sessions


def set_engine(engine: QAEngine) -> None:
    """부팅 시점 교체용. 런타임 중에는 호출하지 않는다."""
    global _engine
    _engine = engine


def allowed_origins() -> list[str]:
    """CORS 허용 출처.

    기본 구성은 FastAPI가 프론트를 같은 출처에서 서빙하므로 **CORS가 필요 없다.**
    별도 개발 서버를 띄울 때만 `configs/*.yaml` 의 `serve.cors_origins` 에 나열한다.
    와일드카드를 기본값으로 두지 않는다.
    """
    return list(get_config().serve.cors_origins)
