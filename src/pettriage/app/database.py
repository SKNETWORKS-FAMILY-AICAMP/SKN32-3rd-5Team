"""MySQL 연결 및 테이블 초기화.

DATABASE_URL 형식: mysql+pymysql://user:password@host:port/dbname
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


@lru_cache(maxsize=1)
def _engine():
    from ..config import get_secrets

    url = get_secrets().database_url
    if not url:
        raise RuntimeError("DATABASE_URL 환경변수가 없습니다. .env 파일을 확인하세요.")
    return create_engine(url, pool_pre_ping=True)


_SessionLocal = sessionmaker(autocommit=False, autoflush=False)


def get_db() -> Generator[Session, None, None]:
    """FastAPI 의존성 주입용 DB 세션."""
    session = _SessionLocal(bind=_engine())
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """앱 기동 시 테이블을 생성한다. 이미 있으면 건드리지 않는다."""
    from . import models  # noqa: F401 — Base.metadata 등록을 위해 반드시 임포트

    Base.metadata.create_all(_engine())
    log.info("DB 테이블 초기화 완료")