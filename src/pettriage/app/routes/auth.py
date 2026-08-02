"""회원가입 · 로그인 라우터.

    POST /api/auth/signup   회원가입
    POST /api/auth/login    로그인 → access_token

설계 근거: docs/06 D-40 · D-36

**스키마를 여기 두지 않는다** — 계약은 `app/contracts.py` 하나다 (D-22 · D-40).
**세션 주입도 `deps` 를 통한다** — 테스트가 `dependency_overrides` 로 갈아끼워야 한다.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..auth import create_access_token, hash_password, verify_password
from ..contracts import LoginRequest, LoginResponse, SignupRequest, SignupResponse
from ..deps import get_db
from ..models import User, utcnow

router = APIRouter(prefix="/api/auth", tags=["auth"])

_db_dep = Depends(get_db)

#: 로그인 실패는 **한 문장으로만** 답한다.
#: "없는 계정"과 "틀린 비밀번호"를 나눠 말하면 그것이 곧 가입 여부 조회 창구가 된다.
_LOGIN_FAILED = "이메일 또는 비밀번호가 올바르지 않습니다."
_EMAIL_TAKEN = "이미 사용 중인 이메일입니다."


@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
def signup(req: SignupRequest, db: Session = _db_dep) -> SignupResponse:
    """회원가입.

    **중복 검사를 조회로만 하지 않는다.** 조회와 삽입 사이에 다른 요청이 끼어들면
    `IntegrityError` 가 500 으로 나간다. `unique=True` 가 데이터는 지켜 주지만
    **응답 코드가 틀리면 프론트가 "서버 장애"로 그린다.**
    최종 판정은 DB 제약이고, 앞의 조회는 흔한 경우를 빨리 걸러내는 용도다.
    """
    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status.HTTP_409_CONFLICT, _EMAIL_TAKEN)

    user = User(
        user_id=str(uuid.uuid4()),
        email=req.email,
        password_hash=hash_password(req.password),
        nickname=req.nickname,
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, _EMAIL_TAKEN) from None
    db.refresh(user)
    return SignupResponse(user_id=user.user_id, nickname=user.nickname)


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest, db: Session = _db_dep) -> LoginResponse:
    """로그인.

    **비활성 계정 검사는 비밀번호 검증 뒤에 한다.** 앞에 두면
    비밀번호를 모르는 사람도 "비활성화된 계정입니다"를 보고 **가입 여부를 알아낸다.**
    """
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, _LOGIN_FAILED)
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "비활성화된 계정입니다.")

    user.last_login_at = utcnow()
    db.commit()

    return LoginResponse(access_token=create_access_token(user.user_id), nickname=user.nickname)
