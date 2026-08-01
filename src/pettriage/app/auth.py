"""비밀번호 해싱 + JWT 유틸리티."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
from passlib.context import CryptContext

from ..config import get_secrets

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── 비밀번호 ──────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


# ── JWT ───────────────────────────────────────────────────

def create_access_token(user_id: str) -> str:
    secrets = get_secrets()
    expire = datetime.now(timezone.utc) + timedelta(minutes=secrets.jwt_expire_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, secrets.jwt_secret_key, algorithm=secrets.jwt_algorithm)


def decode_access_token(token: str) -> str:
    """토큰 검증 후 user_id 반환. 유효하지 않으면 예외 발생."""
    secrets = get_secrets()
    payload = jwt.decode(token, secrets.jwt_secret_key, algorithms=[secrets.jwt_algorithm])
    return payload["sub"]