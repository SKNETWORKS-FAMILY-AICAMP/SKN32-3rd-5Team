"""반려동물 등록 · 조회 라우터.

엔드포인트
  POST /api/pets              반려동물 등록
  GET  /api/pets              내 반려동물 목록
  GET  /api/pets/{pet_id}     반려동물 단건 조회
"""

from __future__ import annotations

import uuid
from datetime import datetime

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from typing import Literal

from ..auth import decode_access_token
from ..database import get_db
from ..models import Pet

router = APIRouter(prefix="/api/pets", tags=["pets"])
_bearer = HTTPBearer()


# ── 현재 사용자 확인 ──────────────────────────────────────

def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> str:
    try:
        return decode_access_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="토큰이 만료되었습니다.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유효하지 않은 토큰입니다.")


# ── 요청/응답 스키마 ──────────────────────────────────────

class PetCreate(BaseModel):
    name: str = Field(min_length=1, max_length=50)
    species: Literal["dog", "cat", "bird"]
    breed: str | None = Field(default=None, max_length=50)
    weight_kg: float | None = Field(default=None, gt=0, le=200)


class PetResponse(BaseModel):
    pet_id: str
    name: str
    species: str
    breed: str | None
    weight_kg: float | None
    created_at: datetime

    class Config:
        from_attributes = True


# ── 엔드포인트 ────────────────────────────────────────────

@router.post("", response_model=PetResponse, status_code=status.HTTP_201_CREATED)
def register_pet(
    req: PetCreate,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
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
def list_pets(
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    return db.query(Pet).filter(Pet.user_id == user_id).all()


@router.get("/{pet_id}", response_model=PetResponse)
def get_pet(
    pet_id: str,
    user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
):
    pet = db.query(Pet).filter(Pet.pet_id == pet_id, Pet.user_id == user_id).first()
    if not pet:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="반려동물을 찾을 수 없습니다.")
    return pet