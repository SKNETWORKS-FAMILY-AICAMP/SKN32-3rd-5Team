"""반려동물 프로필 라우터.

    POST /api/pets            등록
    GET  /api/pets            내 목록
    GET  /api/pets/{pet_id}   단건

설계 근거: docs/06 D-40 · D-36 · D-43

**`jwt` 를 임포트하지 않는다.** 토큰 해석은 `app.auth` 가, 401 번역은 `deps` 가 한다.
배달 계층이 인증 구현체를 알면 라이브러리를 바꿀 때 여기까지 따라 바뀐다.

**모든 조회에 `user_id` 조건이 붙는다.** `pet_id` 만으로 찾으면
남의 반려동물을 ID 추측으로 읽을 수 있다 — `records_store.py` 가
*"pet_id 를 아는 사람은 누구나 그 기록을 읽는다"* 로 경고한 그 구멍이다.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..contracts import PetCreate, PetResponse
from ..deps import get_current_user_id, get_db
from ..models import Pet

router = APIRouter(prefix="/api/pets", tags=["pets"])

_db_dep = Depends(get_db)
_user_dep = Depends(get_current_user_id)


@router.post("", response_model=PetResponse, status_code=status.HTTP_201_CREATED)
def register_pet(
    req: PetCreate,
    user_id: str = _user_dep,
    db: Session = _db_dep,
) -> Pet:
    """등록. **식별자는 앱 내부 UUID 다** — 동물등록번호를 받지 않는다 (D-36 조치 1).

    D-43 사진은 아직 없다. 붙일 때 **관문 5단계를 통과한 것만** 저장한다.
    """
    pet = Pet(
        pet_id=uuid.uuid4().hex,
        user_id=user_id,
        name=req.name,
        species=req.species,
        breed=req.breed,
        weight_kg=req.weight_kg,
    )
    db.add(pet)
    db.commit()
    db.refresh(pet)
    return pet


@router.get("", response_model=list[PetResponse])
def list_pets(user_id: str = _user_dep, db: Session = _db_dep) -> list[Pet]:
    return db.query(Pet).filter(Pet.user_id == user_id).all()


@router.get("/{pet_id}", response_model=PetResponse)
def get_pet(pet_id: str, user_id: str = _user_dep, db: Session = _db_dep) -> Pet:
    """단건 조회.

    **남의 것을 404 로 답한다.** 403 으로 답하면 *"존재는 한다"* 를 알려주게 되어
    ID 를 훑어 다른 사용자의 반려동물 수를 세어 볼 수 있다.
    """
    pet = db.query(Pet).filter(Pet.pet_id == pet_id, Pet.user_id == user_id).first()
    if not pet:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "반려동물을 찾을 수 없습니다.")
    return pet
