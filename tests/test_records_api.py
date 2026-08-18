"""다이어리 기록 API — **DB 저장으로 옮겨 간 뒤의 계약** (2026-08-03).

설계 근거: docs/06 D-36 · D-52 · docs/02 §12

2026-08-03 머지로 `/api/records`·`/api/report` 가 인메모리 `RecordStore` 에서
**MySQL `diary_entries`** 로 옮겨 갔다. 인증이 필수가 되었고 소유자 확인이
**두 단계**가 됐다 — Bearer 토큰으로 `user_id`, 그다음 `(pet_id, user_id)` 매칭
(없으면 남의 것도 404).

그 변경으로 `tests/test_api.py` 의 다이어리 테스트 넷이 **토큰 없이 부르다가**
401 로 끝나 `KeyError: 'timeline'` 을 냈다. **응답 계약이 깨진 것이 아니라
테스트 하네스가 안 맞았다.** 여기로 옮겨 같은 성질을 다시 고정한다.

옮겨 온 성질 —

  · 조류 전용 필드는 종이 맞을 때만 보관한다 (D-36 최소 수집)
  · 기간 필터를 받기만 하고 안 쓰면 **화면의 기간 선택이 거짓말**이 된다

새로 고정하는 성질 — **둘 다 이 머지로 생겼는데 테스트가 없었다** —

  · **남의 반려동물 기록은 404** (D-52). 보안 성질이라 비워 두지 않는다
  · **같은 날짜 재저장은 갱신이다** (하루 1건 정책). 프론트가 "수정" 을 별도 API 없이
    다시 저장으로 처리한다. 이 정책이 조용히 바뀌면 ③ 학습 데이터의 입력
    *(기록 N일치)* 가 함께 흔들린다 (03 §2 · D-83)

`test_auth_api.py` 와 **같은 하네스**다 — SQLite 인메모리로 **DB 서버 없이 라우터까지**
검증한다. `create_app()` 을 쓰지 않는 이유는 위와 같다: DB 오버라이드가 필요하다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy", reason="[db] extra 미설치")
pytest.importorskip("jwt", reason="[db] extra 미설치")
pytest.importorskip("bcrypt", reason="[db] extra 미설치")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from pettriage.app import auth as auth_mod  # noqa: E402
from pettriage.app.deps import get_db  # noqa: E402
from pettriage.app.models import Base  # noqa: E402
from pettriage.app.routes.auth import router as auth_router  # noqa: E402
from pettriage.app.routes.pets import router as pets_router  # noqa: E402
from pettriage.app.routes.records import router as records_router  # noqa: E402

TEST_KEY = "test-key-do-not-use-in-production-0123456789"


@pytest.fixture
def sessionmaker_():
    """파일 없는 SQLite. `StaticPool` 이라야 커넥션 하나를 계속 쓴다."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture
def client(sessionmaker_, monkeypatch):
    monkeypatch.setattr(auth_mod, "_signing_key", lambda: TEST_KEY)

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(pets_router)
    app.include_router(records_router)

    def _db():
        s = sessionmaker_()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _db
    return TestClient(app)


# ── 헬퍼 ─────────────────────────────────────────────────────


def _auth(c, email="a@b.co", nick="한빈") -> dict[str, str]:
    """가입 → 로그인 → Authorization 헤더."""
    c.post("/api/auth/signup", json={"email": email, "password": "hunter22!", "nickname": nick})
    tok = c.post("/api/auth/login", json={"email": email, "password": "hunter22!"}).json()
    return {"Authorization": f"Bearer {tok['access_token']}"}


def _pet(c, headers: dict[str, str], name="코코", species="dog") -> str:
    r = c.post("/api/pets", json={"name": name, "species": species}, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()["pet_id"]


def _record(c, headers: dict[str, str], pet_id: str, species: str, day: str, **kw):
    body = {"pet_id": pet_id, "species": species, "recorded_at": f"{day}T09:00:00", **kw}
    return c.post("/api/records", json=body, headers=headers)


def _timeline(c, headers: dict[str, str], pet_id: str, **params) -> list[dict]:
    r = c.get("/api/report", params={"pet_id": pet_id, **params}, headers=headers)
    assert r.status_code == 200, r.text
    return r.json()["timeline"]


# ── 최소 수집 (D-36) ─────────────────────────────────────────


def test_조류_전용_필드는_포유류에서_버린다(client):
    """`droppings` 는 종이 맞을 때만 보관한다 (D-36).

    ⚠️ 옛 테스트는 `"droppings" not in rows[0]` 였다. DB 행은 칼럼이 늘 있으므로
       **키는 있고 값이 `None`** 이다. 확인할 것은 키의 유무가 아니라 **값이 안 남는 것**이다.
    """
    h = _auth(client)
    pet = _pet(client, h, species="dog")
    _record(client, h, pet, "dog", "2026-07-31", droppings="노란색")

    rows = _timeline(client, h, pet)
    assert rows and rows[0]["droppings"] is None


def test_조류에서는_유지한다(client):
    h = _auth(client)
    pet = _pet(client, h, name="초코", species="bird")
    _record(client, h, pet, "bird", "2026-07-31", droppings="녹색")

    rows = _timeline(client, h, pet)
    assert rows[0]["droppings"] == "녹색"


# ── 기간 필터 ────────────────────────────────────────────────


def test_기간_필터가_실제로_적용된다(client):
    """받기만 하고 안 쓰면 **화면의 기간 선택이 거짓말**이 된다."""
    h = _auth(client)
    pet = _pet(client, h)
    for day in ("2026-07-01", "2026-07-15", "2026-07-30"):
        _record(client, h, pet, "dog", day)

    rows = _timeline(client, h, pet, period_from="2026-07-10", period_to="2026-07-20")
    assert [r["recorded_at"][:10] for r in rows] == ["2026-07-15"]


# ── 소유자 격리 (D-52) ───────────────────────────────────────


def test_토큰_없이는_401(client):
    """인증 정보가 **없는** 것은 401 이다 — 403 은 *"누군지는 안다"* 는 뜻이다."""
    r = client.get("/api/report", params={"pet_id": "whatever"})
    assert r.status_code == 401
    assert r.headers.get("WWW-Authenticate") == "Bearer"


def test_남의_반려동물_기록은_404(client):
    """**남의 것도 404 다** (D-52).

    403 으로 답하면 *"그 pet_id 는 존재한다"* 를 알려 주는 **존재 조회 창구**가 된다.
    `test_auth_api.py::test_없는_계정과_틀린_비밀번호가_같은_말을_한다` 와 같은 이유다.

    ⚠️ 이 성질은 이번 머지로 새로 생겼는데 테스트가 없었다. 보안 성질을 비워 두지 않는다.
    """
    ha = _auth(client, email="a@b.co", nick="한빈")
    pet_a = _pet(client, ha)
    _record(client, ha, pet_a, "dog", "2026-07-31", note="비밀")

    hb = _auth(client, email="b@b.co", nick="다른사람")
    assert client.get("/api/report", params={"pet_id": pet_a}, headers=hb).status_code == 404
    assert _record(client, hb, pet_a, "dog", "2026-08-01").status_code == 404

    # 주인에게는 그대로 보인다 — 404 가 격리가 아니라 고장이어서 난 것이 아님을 확인한다.
    assert len(_timeline(client, ha, pet_a)) == 1


# ── 하루 1건 정책 ────────────────────────────────────────────


def test_같은_날짜_재저장은_갱신이다(client):
    """프론트가 "수정" 을 별도 API 없이 **다시 저장**으로 처리한다 (2026-08-03).

    ⚠️ 이 정책이 조용히 바뀌면 ③ 학습 데이터의 입력 *(기록 N일치)* 가 함께 흔들린다
       (03 §2 · D-83). 하루 여러 건이 쌓이기 시작하면 같은 기간이 다른 길이가 된다.
    """
    h = _auth(client)
    pet = _pet(client, h)

    first = _record(client, h, pet, "dog", "2026-07-31", note="처음")
    second = _record(client, h, pet, "dog", "2026-07-31", note="고침")
    assert first.status_code == second.status_code == 201

    rows = _timeline(client, h, pet)
    assert len(rows) == 1, "같은 날짜가 두 건으로 쌓였다 — 하루 1건 정책이 깨졌다"
    assert rows[0]["note"] == "고침"
    assert first.json()["record_id"] == second.json()["record_id"], "갱신인데 id 가 바뀐다"


def test_다른_날짜는_따로_쌓인다(client):
    """갱신 판정이 `(pet_id, recorded_date)` 인지 확인한다 — pet 단위면 다 덮인다."""
    h = _auth(client)
    pet = _pet(client, h)
    _record(client, h, pet, "dog", "2026-07-30", note="어제")
    _record(client, h, pet, "dog", "2026-07-31", note="오늘")

    rows = _timeline(client, h, pet)
    assert [r["note"] for r in rows] == ["어제", "오늘"]
