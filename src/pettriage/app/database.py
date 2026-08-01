"""관계형 DB 연결 및 테이블 초기화.

`DATABASE_URL` 예:

    mysql+pymysql://user:pw@host:3306/db     운영·개발 (D-48)
    sqlite+pysqlite:///:memory:              테스트

**MySQL 은 팀원 PC 에 설치하지 않는다** — `docker compose up db` 로 띄운다.
버전이 제각각이면 "내 PC에선 되는데" 가 시작된다.

**드라이버를 코드가 고르지 않는다.** SQLAlchemy URL 이 정하므로
테스트는 SQLite 로 같은 코드를 돌린다 — DB 서버 없이 라우터까지 검증된다.
"""

from __future__ import annotations

import logging
from collections.abc import Generator
from functools import lru_cache

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
    """DB 세션 1개. **예외로 빠져나오면 반드시 되감는다.**

    `rollback()` 없이 `close()` 만 하면 **열린 트랜잭션이 붙은 채 커넥션이 풀로 돌아간다.**
    다음 요청이 그 커넥션을 받아 남의 미완결 트랜잭션 위에서 일하게 된다.

    직접 부르지 않는다 — 라우터는 `deps.get_db` 를 통한다 (D-40).
    """
    session = _SessionLocal(bind=_engine())
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db() -> None:
    """앱 기동 시 테이블을 생성한다. 이미 있으면 건드리지 않는다."""
    from . import models  # noqa: F401 — Base.metadata 등록을 위해 반드시 임포트

    Base.metadata.create_all(_engine())
    log.info("DB 테이블 초기화 완료")
