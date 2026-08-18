"""라우터 모음.

DB 라우터(`auth` · `pets` · `users` · **`records`**)는 `DATABASE_URL` 이 설정돼
있을 때만 로드된다. DB 를 안 깐 팀원과 CI 에서도 앱이 뜨게 하기 위한 것이다.

🔴 **`records` 가 이 목록에 늦게 들어왔다 (2026-08-09 수정).** 다이어리가 인메모리
`RecordStore` 에서 **MySQL `diary_entries`** 로 옮겨 가면서(2026-08-03) `records.py` 는
최상단에서 `sqlalchemy.orm.Session` 을 임포트하게 됐다. **성질이 바뀌었는데 분류는
안 바뀌어서** 상시 라우터 자리에 그대로 남았고, `[db]` 없는 구성에서 **앱이 통째로
임포트 실패**했다.

    tests/conftest.py:36  from pettriage.app.main import create_app
    app/routes/__init__.py:36  from .records import router as records_router
    app/routes/records.py:22   from sqlalchemy.orm import Session
    E   ModuleNotFoundError: No module named 'sqlalchemy'

CI 의 `test` 잡(`.[api,dev]`)이 pytest **종료코드 4**(conftest 임포트 실패)로 죽고,
`docker` 잡은 이미지가 `.[api]` 만 깔아 **API 가 기동조차 못 했다.** 두 실패가 한 원인이다.
`db-deps` 잡만 통과한 것이 그 증거였다 — **더 많이 까는 잡만 초록이면 의존성 누락이다.**

> **의존성이 늘어난 모듈은 분류가 바뀐 것이다.** 임포트 한 줄이 늘었을 뿐이라고
> 읽으면 그 모듈이 어느 목록에 있어야 하는지를 다시 묻지 않게 된다 (D-70 · D-48).

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

# .env → os.environ. 쉘 변수가 있으면 덮지 않는다.
# 아래 os.getenv("DATABASE_URL") 체크가 .env 만 있는 구성에서도 참이 되도록.
load_dotenv()

log = logging.getLogger(__name__)


class DBRoutersUnavailableError(RuntimeError):
    """`DATABASE_URL` 은 있는데 DB 라우터를 못 올렸다.

    **조용히 인증 없는 앱으로 넘어가지 않는다.**
    """


#: DB 없이도 도는 것. **여기에 두려면 `[api]` 만으로 임포트돼야 한다.**
_routers = [meta_router, ask_router]

if os.getenv("DATABASE_URL"):
    try:
        from .auth import router as auth_router
        from .pets import router as pets_router
        from .records import router as records_router
        from .users import router as users_router
    except ImportError as e:  # pragma: no cover - 의존성 미설치 경로
        raise DBRoutersUnavailableError(
            f"DATABASE_URL 이 설정됐는데 DB 라우터를 못 올렸다: {e}\n"
            "  → pip install -e '.[api,db]' -c constraints.txt\n"
            "  DB 없이 띄우려면 DATABASE_URL 을 지운다."
        ) from e
    _routers.extend([auth_router, pets_router, users_router, records_router])
else:
    log.info(
        "DATABASE_URL 미설정 — auth/pets/users/records 라우터 비활성화 (의도된 구성). "
        "다이어리는 DB 저장이므로 이 구성에서는 뜨지 않는다"
    )

ALL_ROUTERS = tuple(_routers)

# `records_router` 는 조건부라 여기에 없다 — 없을 수 있는 이름을 `__all__` 에 두면
# `from ... import *` 가 구성에 따라 깨진다. 필요하면 모듈에서 직접 가져온다:
#     from pettriage.app.routes.records import router
__all__ = [
    "ALL_ROUTERS",
    "DBRoutersUnavailableError",
    "ask_router",
    "meta_router",
]
