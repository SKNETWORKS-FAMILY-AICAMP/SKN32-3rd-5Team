"""FastAPI 앱 — 배달 계층 (05 §2 조각 12).

    실행:  make serve      →  http://127.0.0.1:8000
    문서:  http://127.0.0.1:8000/docs   (OpenAPI 스펙 = WS2·WS5 합의문)

정적 프론트를 같은 출처에서 서빙한다. 빌드 도구도 CORS 설정도 없다 —
시연 재현이 `make serve` 한 줄로 끝나는 쪽을 택했다 (04 §8 재현성).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from .deps import allowed_origins
from .routes import ALL_ROUTERS

log = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parents[3] / "web"

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


def create_app() -> FastAPI:
    app = FastAPI(
        title="PetTriage API",
        version=__version__,
        description=DESCRIPTION,
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

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")
    else:  # pragma: no cover
        log.warning("web/ 디렉터리가 없다 — API만 제공한다 (%s)", WEB_DIR)

    return app


app = create_app()


def run() -> None:
    """`pettriage-serve` 진입점. 호스트·포트는 설정에서 온다."""
    import uvicorn

    from ..config import get_config

    cfg = get_config().serve
    uvicorn.run("pettriage.app.main:app", host=cfg.host, port=cfg.port, reload=False)
