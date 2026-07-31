"""FastAPI 앱 — 배달 계층 (05 §2 조각 12).

    실행:  make serve      →  http://127.0.0.1:8000
    문서:  http://127.0.0.1:8000/docs   (OpenAPI 스펙 = WS2·WS5 합의문)

정적 프론트를 같은 출처에서 서빙한다. 빌드 도구도 CORS 설정도 없다 —
시연 재현이 `make serve` 한 줄로 끝나는 쪽을 택했다 (04 §8 재현성).
"""

from __future__ import annotations

import logging

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


def create_app() -> FastAPI:
    app = FastAPI(title="PetTriage API", version=__version__, description=DESCRIPTION)

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
