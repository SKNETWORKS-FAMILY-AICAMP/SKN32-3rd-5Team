"""라우터 모음.

DB 관련 라우터(auth, pets)는 `DATABASE_URL` 이 설정돼 있을 때만 로드된다.
DB 미설치 팀원 · CI 에서도 앱이 뜬다.
"""

from __future__ import annotations

import logging
import os

from .ask import router as ask_router
from .meta import router as meta_router
from .records import router as records_router

log = logging.getLogger(__name__)

_routers = [meta_router, ask_router, records_router]

# DB 라우터는 선택 로드 — DATABASE_URL 없으면 건너뛴다.
if os.getenv("DATABASE_URL"):
    try:
        from .auth import router as auth_router
        from .pets import router as pets_router

        _routers.extend([auth_router, pets_router])
    except ImportError as e:
        log.warning("DB 라우터 로드 실패 — auth/pets 비활성화: %s", e)
else:
    log.info("DATABASE_URL 미설정 — auth/pets 라우터 비활성화")

ALL_ROUTERS = tuple(_routers)

__all__ = ["ALL_ROUTERS"]