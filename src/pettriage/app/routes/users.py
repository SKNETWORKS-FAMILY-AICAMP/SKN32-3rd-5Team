"""사용자 정보 라우터.

    GET /api/users/me       현재 로그인 사용자 정보

설계 근거: docs/06 D-40 · D-36 · D-48

**JWT 토큰의 `user_id` 로만 조회한다.** URL·body 로 user_id 받지 않는다 —
받으면 남의 정보를 볼 수 있게 된다 (D-52 후속: `pets.py` 와 같은 원칙).

**스키마는 `contracts.py` 하나다** (D-40). 여기서 재정의하지 않는다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..contracts import UserMeResponse
from ..deps import get_current_user_id, get_db
from ..models import User

router = APIRouter(prefix="/api/users", tags=["users"])

_db_dep = Depends(get_db)
_user_dep = Depends(get_current_user_id)


@router.get("/me", response_model=UserMeResponse)
def get_me(user_id: str = _user_dep, db: Session = _db_dep) -> User:
    """현재 로그인 사용자의 최소 정보.

    확장 시 유의: 민감 필드(is_active·last_login_at 등)를 추가할 때
    **UserMeResponse 스키마와 함께** 검토한다 — 응답 필드는 스키마가 강제한다.
    """
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        # 토큰은 유효한데 DB 에 사용자가 없다 — 삭제된 계정 등.
        raise HTTPException(status.HTTP_404_NOT_FOUND, "사용자를 찾을 수 없습니다.")
    return user