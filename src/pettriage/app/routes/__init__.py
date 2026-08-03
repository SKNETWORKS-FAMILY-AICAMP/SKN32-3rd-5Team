"""라우터 모음.

DB 라우터(`auth` · `pets`)는 `DATABASE_URL` 이 설정돼 있을 때만 로드된다.
DB 를 안 깐 팀원과 CI 에서도 앱이 뜨게 하기 위한 것이다.

**두 경우를 구분한다.** 예전에는 하나로 묶여 있었고, 그래서 사고가 났다.

| 상황 | 처리 | 왜 |
|---|---|---|
| `DATABASE_URL` 없음 | 조용히 건너뛴다 | **의도된 것이다.** DB 없이 RAG만 쓰는 구성이 정상이다 |
| `DATABASE_URL` 있는데 임포트 실패 | **기동 중단** | 의도가 아니다. 인증을 켜려던
  사람에게 안 켜졌다고 말해야 한다 |

    아래는 2026-08-01 PR#3 검수에서 실제로 관측된 것이다.

        WARNING: DB 라우터 로드 실패 — auth/pets 비활성화: No module named 'jwt'
        라우터 3 개

    **앱이 정상 기동했다. 회원가입도 로그인도 없는 채로.**
    `passlib`·`PyJWT`·`PyMySQL` 이 `pyproject.toml` 에 선언돼 있지 않아
    저장소를 받은 사람에게는 인증이 통째로 빠졌는데,
    단서는 아무도 안 읽는 `WARNING` 한 줄뿐이었다.

    04 §8 — **검사 축소는 드러나야 한다.** 기능 축소도 마찬가지다.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

from .ask import router as ask_router
from .meta import router as meta_router
from .records import router as records_router

# .env → os.environ. 쉘 변수가 있으면 덮지 않는다.
# 아래 os.getenv("DATABASE_URL") 체크가 .env 만 있는 구성에서도 참이 되도록.
load_dotenv()

log = logging.getLogger(__name__)


class DBRoutersUnavailableError(RuntimeError):
    """`DATABASE_URL` 은 있는데 DB 라우터를 못 올렸다.

    **조용히 인증 없는 앱으로 넘어가지 않는다.**
    """


_routers = [meta_router, ask_router, records_router]

if os.getenv("DATABASE_URL"):
    try:
        from .auth import router as auth_router
        from .pets import router as pets_router
        from .users import router as users_router
    except ImportError as e:  # pragma: no cover - 의존성 미설치 경로
        raise DBRoutersUnavailableError(
            f"DATABASE_URL 이 설정됐는데 DB 라우터를 못 올렸다: {e}\n"
            "  → pip install -e '.[api,db]' -c constraints.txt\n"
            "  DB 없이 띄우려면 DATABASE_URL 을 지운다."
        ) from e
    _routers.extend([auth_router, pets_router, users_router])
else:
    log.info("DATABASE_URL 미설정 — auth/pets/users 라우터 비활성화 (의도된 구성)")

ALL_ROUTERS = tuple(_routers)

__all__ = [
    "ALL_ROUTERS",
    "DBRoutersUnavailableError",
    "ask_router",
    "meta_router",
    "records_router",
]
