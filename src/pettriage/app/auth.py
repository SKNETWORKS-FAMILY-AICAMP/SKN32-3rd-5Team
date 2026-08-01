"""비밀번호 해싱 + JWT.

설계 근거: docs/06 D-40 · D-41 · D-36

**서명 키에 기본값을 두지 않는다.** 자리표시자를 기본값으로 깔면 아무도 안 바꾼 채
배포되고, 키를 아는 사람은 누구나 토큰을 위조한다. 없으면 여기서 **명시적으로 실패한다** —
조용히 약한 키로 도는 것보다 낫다.

**라우터가 `jwt` 를 직접 임포트하지 않게 한다** (D-40 계층 분리).
처음 들어왔을 때 `routes/pets.py` 가 `jwt.ExpiredSignatureError` 를 잡고 있었다.
배달 계층이 인증 구현체를 알면 라이브러리를 바꿀 때 라우터까지 따라 바뀐다.
여기서 우리 예외 두 종으로 번역한다 (2026-08-01 흡수).

**만료 시간·알고리즘은 비밀이 아니라 파라미터다** (D-41) — `configs/*.yaml` 의 `auth` 절에 둔다.
`.env` 에는 키만 남긴다.

**`passlib` 을 쓰지 않는다.** 처음 들어왔을 때 `CryptContext(schemes=["bcrypt"])` 였는데,
`passlib 1.7.4` (2020년 마지막 릴리스) 가 `bcrypt>=4.1` 과 깨져 있다.

    AttributeError: module 'bcrypt' has no attribute '__about__'
    → ValueError: password cannot be longer than 72 bytes

    **9바이트짜리 비밀번호에서도 터진다.** 버전 탐지에 실패한 passlib 이
    길이 검사 분기를 잘못 타기 때문이다. 회원가입이 **한 건도 되지 않는다.**
    2026-08-01 흡수 과정에서 테스트를 붙이자마자 드러났다.

핀을 내려 맞추는 대신 **유지보수가 끊긴 의존성을 걷어냈다.** `bcrypt` 를 직접 쓰면
API 가 두 줄이고, 72바이트 절단도 우리가 명시적으로 막을 수 있다.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import bcrypt
import jwt

from ..config import get_config, get_secrets

#: bcrypt 가 받는 비밀번호 상한. **넘는 부분은 조용히 버려진다** — 그래서 직접 막는다.
BCRYPT_MAX_BYTES = 72


class JWTKeyMissingError(RuntimeError):
    """서명 키가 없다. **약한 기본 키로 넘어가지 않는다.**"""


class TokenExpiredError(Exception):
    """토큰 유효기간이 지났다."""


class TokenInvalidError(Exception):
    """서명·형식이 잘못됐다. **원인을 사용자에게 알리지 않는다** — 키 탐색의 단서가 된다."""


# ── 비밀번호 ──────────────────────────────────────────────


class PasswordTooLongError(ValueError):
    """bcrypt 상한 초과. **조용히 자르지 않는다.**

    자르면 사용자는 긴 비밀번호를 썼다고 믿는데 실제로는 앞 72바이트만 쓰인다.
    한글은 3바이트/자라 25자부터 뒤가 무시된다.
    """


def _encode(plain: str) -> bytes:
    raw = plain.encode("utf-8")
    if len(raw) > BCRYPT_MAX_BYTES:
        raise PasswordTooLongError(
            f"비밀번호가 {len(raw)}바이트다. bcrypt 상한은 {BCRYPT_MAX_BYTES}바이트 "
            "(한글은 3바이트/자)."
        )
    return raw


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(_encode(plain), bcrypt.gensalt()).decode("ascii")


def verify_password(plain: str, hashed: str) -> bool:
    """**틀린 비밀번호와 깨진 해시를 같은 `False` 로 답한다.**

    저장된 해시가 손상됐을 때 예외가 올라가면 500 이 나가고,
    그 자체가 "이 계정은 존재한다"는 신호가 된다.
    """
    try:
        return bcrypt.checkpw(plain.encode("utf-8")[:BCRYPT_MAX_BYTES], hashed.encode("ascii"))
    except (ValueError, UnicodeError):
        return False


# ── JWT ───────────────────────────────────────────────────


def _signing_key() -> str:
    key = get_secrets().jwt_secret_key
    if key is None or not key.get_secret_value().strip():
        raise JWTKeyMissingError(
            "JWT_SECRET_KEY 가 없다. .env 에 넣을 것 — "
            'python -c "import secrets; print(secrets.token_urlsafe(32))"'
        )
    return key.get_secret_value()


def create_access_token(user_id: str) -> str:
    cfg = get_config().auth
    expire = datetime.now(UTC) + timedelta(minutes=cfg.expire_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, _signing_key(), algorithm=cfg.algorithm)


def decode_access_token(token: str) -> str:
    """토큰 검증 후 `user_id` 반환.

    Raises:
        TokenExpiredError: 유효기간 만료
        TokenInvalidError: 서명·형식 오류
    """
    try:
        payload = jwt.decode(token, _signing_key(), algorithms=[get_config().auth.algorithm])
    except jwt.ExpiredSignatureError as e:
        raise TokenExpiredError from e
    except jwt.InvalidTokenError as e:
        raise TokenInvalidError from e
    return payload["sub"]
