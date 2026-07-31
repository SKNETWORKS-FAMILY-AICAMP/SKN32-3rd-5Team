"""라우터 모음."""

from .ask import router as ask_router
from .meta import router as meta_router
from .records import router as records_router

ALL_ROUTERS = (meta_router, ask_router, records_router)

__all__ = ["ALL_ROUTERS", "ask_router", "meta_router", "records_router"]
