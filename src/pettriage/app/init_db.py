"""DB 테이블 초기화 스크립트.

실행:  python -m pettriage.app.init_db
"""

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    from .database import init_db

    log.info("DB 테이블 생성 시작...")
    init_db()
    log.info("완료.")


if __name__ == "__main__":
    main()