"""FastAPI 앱 — 배달 계층 (05 §2 조각 12).

    실행:  make serve      →  http://127.0.0.1:8000
    문서:  http://127.0.0.1:8000/docs   (OpenAPI 스펙 = WS2·WS5 합의문)

정적 프론트를 같은 출처에서 서빙한다. 빌드 도구도 CORS 설정도 없다 —
시연 재현이 `make serve` 한 줄로 끝나는 쪽을 택했다 (04 §8 재현성).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import ResponseValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__, paths
from .contracts import DISCLAIMER
from .deps import allowed_origins
from .routes import ALL_ROUTERS

log = logging.getLogger(__name__)

DESCRIPTION = """
반려동물 헬스케어 다이어리 & 응급 대응 시스템의 배달 계층.

**응답 규약** — `/api/ask` 는 항상 200을 반환하고 `status` 로 분기한다.

| status | 의미 | 화면 |
|---|---|---|
| `answered` | 근거를 찾아 판정했다 | 트리아지 배지 + 근거 |
| `clarify` | 슬롯이 비어 되묻는다 (최대 2회) | 되묻기 대화 |
| `refused` | 근거 없음·판정 불가 | 거절 + 수의사 상담 권고 |

`answered` 응답은 **근거(`citations`)가 비면 생성 자체가 불가능**하다.
모든 응답에 `disclaimer` 가 실린다.
"""


def _install_response_guard(app: FastAPI) -> None:
    """응답 검증 실패를 **거절 화면**으로 돌린다.

    계약 불변식(근거 없는 `answered` 등)이 깨지면 FastAPI는 500을 낸다.
    500은 프론트에서 장애 화면으로 그려지는데, 02 §9는 이런 경우에도
    사용자에게 **행동 지시**를 주라고 정한다. 그래서 200 + `refused` 로 내린다.

    다만 이것은 **버그를 숨기는 것이 아니다** — 로그에는 ERROR로 남고,
    검증 실패 상세에 입력 원문이 실릴 수 있으므로 메시지는 생략한다 (D-36).
    """

    @app.exception_handler(ResponseValidationError)
    async def _on_response_invalid(request: Request, exc: ResponseValidationError):
        log.error(
            "응답 계약 위반 — path=%s errors=%d (상세는 개인정보 우려로 생략)",
            request.url.path,
            len(exc.errors()),
        )
        return JSONResponse(
            status_code=200,
            content={
                "status": "refused",
                "session_id": "",
                "answer": None,
                "triage": None,
                "citations": [],
                "clarify": None,
                "refusal": {
                    "reason": "판정불가",
                    "message": "안전 조건을 만족하는 답변을 만들지 못했습니다.",
                    "advice": "수의사와 상담하시기 바랍니다.",
                },
                "disclaimer": DISCLAIMER,
                "full_text": (
                    "안전 조건을 만족하는 답변을 만들지 못했습니다. "
                    "수의사와 상담하시기 바랍니다. " + DISCLAIMER
                ),
            },
        )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """기동 시 **엔진을 만들고** 임베딩 모델을 미리 올린다 (D-53).

    ## ① 엔진을 여기서 만든다 — 실패를 첫 요청까지 미루지 않는다

    `get_engine()` 은 원래 첫 요청에서 lazy 로 불렸다. 그래서
    `serve.engine=graph` 인데 노드가 비어 있으면 **서버는 조용히 뜨고 첫 사용자가 500 을 받았다.**
    `EngineUnavailable` 은 *"스텁으로 내려가면 평가가 오염된다"* 를 위해 일부러 크게 실패하는
    예외인데(04 §8), 그 실패가 기동이 아니라 사용자에게 도착하면 의미가 없다.

    ## ② `cfg.serve.engine` 이 아니라 **실제 엔진 이름**을 본다

    `PETTRIAGE_ALLOW_ENGINE_FALLBACK=1` 이면 `graph` 설정이어도 스텁으로 내려간다.
    설정만 보고 워밍업하면 **쓰지도 않을 모델을 올린다.**

    ## ③ 워밍업은 기동을 막지 않는다

    모델을 못 받는 환경(오프라인 CI · GPU 없는 팀원)에서도 API 계약과 화면은 돌아야 한다.
    실패하면 로그에 남기고 `/api/health` 의 `model_loaded=false` 로 드러낸다 —
    **조용히 넘어가지 않는다.**
    """
    import os

    from ..config import get_config
    from .deps import get_engine

    # ── DB 스키마 자동 생성 (2026-08-03 개정) ─────────────────
    # 팀원 온보딩 편의 — `python -m pettriage.app.database` 를 매번 부르지 않게 한다.
    # `create_all` 은 **이미 있는 테이블은 건드리지 않는다** — 재기동이 안전하다.
    # 프로덕션에 붙일 때는 프로파일 체크나 마이그레이션 도구 (Alembic 등) 로 교체한다.
    if os.getenv("DATABASE_URL"):
        try:
            from .database import init_db

            init_db()
        except Exception as e:  # noqa: BLE001
            log.error(
                "DB 스키마 초기화 실패 — %s: %s (DATABASE_URL 접속 확인 필요)",
                type(e).__name__,
                e,
            )
            raise

    cfg = get_config()
    engine = get_engine()  # EngineUnavailable 이면 여기서 크게 실패한다 (의도)

    if engine.name == "stub":
        log.info("워밍업 생략 — StubEngine 은 벡터 검색을 하지 않는다")
    elif not cfg.serve.warmup:
        log.warning("워밍업 꺼짐 (serve.warmup=false) — 첫 질의가 모델 로딩을 맞는다")
    else:
        try:
            from ..retrieval.embedder import warm_up

            took = warm_up(cfg.retrieval.embedding_model)
            if took is None:
                log.info(
                    "워밍업 불필요 — 로드가 필요 없는 임베더다 (%s)",
                    cfg.retrieval.embedding_model,
                )
            else:
                log.info("임베딩 워밍업 완료 — %.1fs", took)
        except Exception as e:  # noqa: BLE001 — 기동을 막지 않는다
            log.warning(
                "임베딩 워밍업 실패 (%s) — 첫 질의가 로딩을 맞는다. "
                "/api/health 의 model_loaded 를 볼 것",
                type(e).__name__,
            )
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="PetTriage API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=_lifespan,
    )

    origins = allowed_origins()
    if origins:
        from fastapi.middleware.cors import CORSMiddleware

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )

    for router in ALL_ROUTERS:
        app.include_router(router)

    _install_response_guard(app)

    web = paths.web_dir()
    if web is not None:
        app.mount("/static", StaticFiles(directory=web), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(web / "index.html")
    else:  # pragma: no cover
        log.warning("web/ 디렉터리를 찾지 못했다 — API만 제공한다.")

    return app


app = create_app()


def run() -> None:
    """`pettriage-serve` 진입점. 호스트·포트는 설정에서 온다."""
    import uvicorn

    from ..config import get_config

    cfg = get_config().serve
    uvicorn.run("pettriage.app.main:app", host=cfg.host, port=cfg.port, reload=False)
