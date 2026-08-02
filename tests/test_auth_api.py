"""계정 · 반려동물 프로필 API.

설계 근거: docs/06 D-36 · D-40 · docs/04 §8

**이 테스트가 없으면 이 코드는 CI 에서 한 줄도 실행되지 않는다.**
CI 에는 `DATABASE_URL` 이 없어 라우터가 임포트조차 되지 않고, `ruff` 는 문법만 본다.
2026-08-01 PR#3 검수 시점에 424줄이 그 상태로 들어와 있었다.

SQLite 인메모리로 돌린다 — **DB 서버 없이 라우터까지 통째로** 검증된다.
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
from pettriage.app.models import Base, User  # noqa: E402
from pettriage.app.routes.auth import router as auth_router  # noqa: E402
from pettriage.app.routes.pets import router as pets_router  # noqa: E402

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

    def _db():
        s = sessionmaker_()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = _db
    return TestClient(app)


def _signup(c, email="a@b.co", pw="hunter22!", nick="한빈"):
    return c.post("/api/auth/signup", json={"email": email, "password": pw, "nickname": nick})


def _login(c, email="a@b.co", pw="hunter22!"):
    return c.post("/api/auth/login", json={"email": email, "password": pw})


class TestSignup:
    def test_가입_성공(self, client):
        r = _signup(client)
        assert r.status_code == 201
        assert r.json()["nickname"] == "한빈"

    def test_이메일_중복은_409(self, client):
        """**500 이 아니다.** 프론트가 "서버 장애"로 그리면 안 된다."""
        _signup(client)
        assert _signup(client).status_code == 409

    @pytest.mark.parametrize("email", ["a@", "abc", "@b.co", ""])
    def test_이메일_형식_검증(self, client, email):
        """`str` 로 두면 `"a@"` 도 통과했다."""
        assert _signup(client, email=email).status_code == 422

    def test_짧은_비밀번호_거부(self, client):
        assert _signup(client, pw="1234567").status_code == 422

    def test_긴_비밀번호_거부(self, client):
        """bcrypt 는 **72바이트 초과분을 조용히 버린다.**

        한글은 3바이트/자라 25자부터 뒤가 무시된다. 사용자가 긴 비밀번호를 썼다고
        믿는 채 앞부분만 쓰이는 상황을 만들지 않는다 — 상한을 걸어 명시적으로 막는다.
        """
        assert _signup(client, pw="가" * 65).status_code == 422

    def test_비밀번호를_평문으로_저장하지_않는다(self, client, sessionmaker_):
        _signup(client)
        with sessionmaker_() as s:
            u = s.query(User).first()
        assert u.password_hash != "hunter22!"
        assert u.password_hash.startswith("$2")


class TestLogin:
    def test_로그인_성공(self, client):
        _signup(client)
        r = _login(client)
        assert r.status_code == 200
        assert r.json()["token_type"] == "bearer"

    def test_없는_계정과_틀린_비밀번호가_같은_말을_한다(self, client):
        """나눠 말하면 **가입 여부 조회 창구**가 된다."""
        _signup(client)
        a = _login(client, email="nobody@b.co")
        b = _login(client, pw="wrongpass1")
        assert a.status_code == b.status_code == 401
        assert a.json()["detail"] == b.json()["detail"]

    def test_비활성_계정은_403(self, client, sessionmaker_):
        _signup(client)
        with sessionmaker_() as s:
            s.query(User).update({User.is_active: False})
            s.commit()
        assert _login(client).status_code == 403

    def test_비활성_판정이_비밀번호_뒤에_온다(self, client, sessionmaker_):
        """앞에 두면 비밀번호를 모르는 사람이 **계정 존재를 알아낸다.**"""
        _signup(client)
        with sessionmaker_() as s:
            s.query(User).update({User.is_active: False})
            s.commit()
        assert _login(client, pw="wrongpass1").status_code == 401  # 403 이면 유출

    def test_last_login_at_이_채워진다(self, client, sessionmaker_):
        """스키마에만 있고 아무도 안 채우는 죽은 칼럼이었다."""
        _signup(client)
        _login(client)
        with sessionmaker_() as s:
            assert s.query(User).first().last_login_at is not None


class TestToken:
    def test_왕복(self, client, monkeypatch):
        monkeypatch.setattr(auth_mod, "_signing_key", lambda: TEST_KEY)
        tok = auth_mod.create_access_token("u-1")
        assert auth_mod.decode_access_token(tok) == "u-1"

    def test_서명이_다르면_거부(self, client, monkeypatch):
        monkeypatch.setattr(auth_mod, "_signing_key", lambda: TEST_KEY)
        tok = auth_mod.create_access_token("u-1")
        monkeypatch.setattr(auth_mod, "_signing_key", lambda: "다른-키-다른-키-다른-키-01234567")
        with pytest.raises(auth_mod.TokenInvalidError):
            auth_mod.decode_access_token(tok)

    def test_만료_토큰은_TokenExpiredError(self, client, monkeypatch):
        import jwt as pyjwt

        monkeypatch.setattr(auth_mod, "_signing_key", lambda: TEST_KEY)
        expired = pyjwt.encode({"sub": "u-1", "exp": 1}, TEST_KEY, algorithm="HS256")
        with pytest.raises(auth_mod.TokenExpiredError):
            auth_mod.decode_access_token(expired)

    def test_키가_없으면_명시적으로_실패한다(self, monkeypatch):
        """**약한 기본 키로 조용히 도는 것**을 막는 테스트다.

        `"change-me-in-production"` 이 기본값으로 있던 자리다.
        """
        from pettriage.config import Secrets

        monkeypatch.setattr("pettriage.app.auth.get_secrets", lambda: Secrets(jwt_secret_key=None))
        with pytest.raises(auth_mod.JWTKeyMissingError):
            auth_mod.create_access_token("u-1")

    def test_라우터가_jwt를_직접_모른다(self):
        """D-40 — 배달 계층이 인증 구현체를 알면 라이브러리 교체가 여기까지 번진다."""
        from pathlib import Path

        import pettriage.app.routes.pets as pets_mod

        src = Path(pets_mod.__file__).read_text(encoding="utf-8")
        assert "import jwt" not in src


class TestPets:
    def _auth(self, client):
        _signup(client)
        return {"Authorization": f"Bearer {_login(client).json()['access_token']}"}

    def test_토큰_없이는_401(self, client):
        """인증 정보가 **없는** 것은 401(Unauthorized) 이다.

        ⚠️ 2026-08-02에 `403 → 401` 로 바뀌었다. 코드가 아니라 **테스트가 틀렸다.**
        `deps.get_current_user_id` 는 주석부터 *"실패하면 401"* 이라고 적고 있고,
        `HTTPBearer(auto_error=True)` 의 상태 코드가 FastAPI 0.13x 에서 403→401 로
        바뀌면서 드러났다. 의미상으로도 401 이 맞다 —
        403 은 "누군지 알지만 권한이 없다" 이고, 여기는 누군지조차 모른다.
        """
        r = client.post("/api/pets", json={"name": "코코", "species": "dog"})
        assert r.status_code == 401

    def test_잘못된_토큰은_401(self, client):
        r = client.get("/api/pets", headers={"Authorization": "Bearer not-a-real-token"})
        assert r.status_code == 401

    def test_등록과_조회(self, client):
        h = self._auth(client)
        r = client.post("/api/pets", json={"name": "코코", "species": "bird"}, headers=h)
        assert r.status_code == 201
        pid = r.json()["pet_id"]
        assert client.get(f"/api/pets/{pid}", headers=h).json()["name"] == "코코"

    def test_종은_3종으로_제한된다(self, client):
        h = self._auth(client)
        r = client.post("/api/pets", json={"name": "x", "species": "hamster"}, headers=h)
        assert r.status_code == 422

    def test_남의_반려동물은_404(self, client):
        """**403 이 아니다** — 403 은 "존재는 한다"를 알려준다."""
        h1 = self._auth(client)
        pid = client.post("/api/pets", json={"name": "코코", "species": "cat"}, headers=h1).json()[
            "pet_id"
        ]

        _signup(client, email="c@d.co")
        tok2 = _login(client, email="c@d.co").json()["access_token"]
        h2 = {"Authorization": f"Bearer {tok2}"}
        assert client.get(f"/api/pets/{pid}", headers=h2).status_code == 404
        assert client.get("/api/pets", headers=h2).json() == []

    def test_체중_범위(self, client):
        h = self._auth(client)
        for w in (0, -1, 201):
            r = client.post(
                "/api/pets", json={"name": "x", "species": "dog", "weight_kg": w}, headers=h
            )
            assert r.status_code == 422, w


class TestPrivacy:
    """D-36 — 최소 수집."""

    def test_되묻기_슬롯이_테이블에_없다(self):
        """`clarify_turns`·`weight_kg`·`amount_g` 는 휘발이 의도다 (05 §3).

        되살리려면 D-36 을 뒤집는 결정이므로 06 에 기록부터 해야 한다.
        """
        from pettriage.app.models import ChatSession

        cols = set(ChatSession.__table__.columns.keys())
        assert cols.isdisjoint({"clarify_turns", "weight_kg", "amount_g"}), cols

    def test_동물등록번호를_받지_않는다(self):
        """등록번호에는 소유자 성명·주민등록번호·주소·전화번호가 묶여 있다 (D-36 조치 1)."""
        from pettriage.app.contracts import PetCreate

        assert not {"registration_no", "animal_id", "등록번호"} & set(PetCreate.model_fields)

    def test_시각이_timezone_aware_다(self):
        """naive 로 저장하면 KST 변환을 한 번만 잊어도 9시간 틀린다."""
        from pettriage.app.models import utcnow

        assert utcnow().tzinfo is not None
