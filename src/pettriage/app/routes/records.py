"""POST /api/records · GET /api/report — 다이어리 (02 §12).

**MySQL `diary_entries` 테이블에 영구 저장한다** (2026-08-03 개정).
예전에는 인메모리 `RecordStore` 였다. 서버가 재시작될 때마다 기록이 다 날아가서
데모에는 됐지만 실사용은 안 됐다.

소유자 확인은 **두 단계**로 한다 (D-52).
  1. Bearer 토큰 → `user_id`
  2. `(pet_id, user_id)` 매칭되는 pet 이 있는지 — 없으면 404 (남의 것도 404)

⚠️ **요약은 기록 원문을 모델에 보낸다.** D-18(가상 프로필만 넣는다)이 지켜지는
   동안에만 안전하고, `privacy/` 필터가 붙기 전에는 실입력을 태우지 않는다 (D-36).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from ..contracts import RecordCreate, RecordCreated, ReportResponse
from ..deps import get_current_user_id, get_db
from ..models import DiaryEntry, Pet
from ..report import summarize_period

router = APIRouter(prefix="/api", tags=["records"])

_db_dep = Depends(get_db)
_user_dep = Depends(get_current_user_id)


def _row_to_dict(e: DiaryEntry) -> dict:
    """DB 행 → 프론트가 기대하는 timeline row (RecordCreate 를 model_dump 한 모양)."""
    return {
        "record_id": e.entry_id,
        "pet_id": e.pet_id,
        "recorded_at": e.recorded_at.isoformat() if e.recorded_at else "",
        "note": e.note or "",
        "meals": json.loads(e.meals) if e.meals else [],
        "symptoms": json.loads(e.symptoms) if e.symptoms else [],
        "weight_kg": e.weight_kg,
        "droppings": e.droppings,
    }


@router.post("/records", response_model=RecordCreated, status_code=status.HTTP_201_CREATED)
def create_record(
    rec: RecordCreate,
    user_id: str = _user_dep,
    db: Session = _db_dep,
) -> RecordCreated:
    """다이어리 기록 저장 (upsert).

    **같은 `(pet_id, recorded_date)` 가 이미 있으면 갱신한다** (2026-08-03).
    프론트가 "수정" 을 별도 API 없이 다시 저장으로 처리한다 — 하루 1건 정책.
    소유자 확인은 pet 조회 시 붙는다 (D-52).
    """
    pet = db.query(Pet).filter(Pet.pet_id == rec.pet_id, Pet.user_id == user_id).first()
    if not pet:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "반려동물을 찾을 수 없습니다.")

    # RecordCreate.recorded_at 은 이미 ISO 8601 로 정규화돼 있다 (contracts._normalize_recorded_at).
    recorded_at = datetime.fromisoformat(rec.recorded_at)
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=UTC)

    # 조류가 아니면 droppings 는 버린다 (D-36 최소 수집).
    droppings = rec.droppings if rec.species == "bird" else None
    meals_json = json.dumps(rec.meals, ensure_ascii=False)
    symptoms_json = json.dumps(rec.symptoms, ensure_ascii=False)

    # 같은 날짜 기록이 있으면 그것을 갱신 — record_id 는 유지된다.
    existing = (
        db.query(DiaryEntry)
        .filter(
            DiaryEntry.pet_id == rec.pet_id,
            DiaryEntry.user_id == user_id,
            DiaryEntry.recorded_date == recorded_at.date(),
        )
        .first()
    )
    if existing:
        existing.recorded_at = recorded_at
        existing.weight_kg = rec.weight_kg
        existing.meals = meals_json
        existing.symptoms = symptoms_json
        existing.note = rec.note or None
        existing.droppings = droppings
        db.commit()
        db.refresh(existing)
        return RecordCreated(record_id=existing.entry_id, pet_id=existing.pet_id, indexed=False)

    entry = DiaryEntry(
        entry_id=uuid4().hex[:12],
        pet_id=rec.pet_id,
        user_id=user_id,
        recorded_date=recorded_at.date(),
        recorded_at=recorded_at,
        weight_kg=rec.weight_kg,
        meals=meals_json,
        symptoms=symptoms_json,
        note=rec.note or None,
        droppings=droppings,
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)

    return RecordCreated(record_id=entry.entry_id, pet_id=entry.pet_id, indexed=False)


@router.get("/report", response_model=ReportResponse)
def report(
    pet_id: str = Query(max_length=64),
    period_from: str = Query(default=""),
    period_to: str = Query(default=""),
    user_id: str = _user_dep,
    db: Session = _db_dep,
) -> ReportResponse:
    """기간 리포트. 소유자 확인 후 DB 에서 조회 → 요약."""
    pet = db.query(Pet).filter(Pet.pet_id == pet_id, Pet.user_id == user_id).first()
    if not pet:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "반려동물을 찾을 수 없습니다.")

    q = (
        db.query(DiaryEntry)
        .filter(DiaryEntry.pet_id == pet_id, DiaryEntry.user_id == user_id)
    )
    if period_from:
        q = q.filter(DiaryEntry.recorded_date >= datetime.fromisoformat(period_from).date())
    if period_to:
        q = q.filter(DiaryEntry.recorded_date <= datetime.fromisoformat(period_to).date())
    entries = q.order_by(DiaryEntry.recorded_at).all()

    rows = [_row_to_dict(e) for e in entries]
    summary, summary_by = summarize_period(rows, period_from, period_to)
    return ReportResponse(
        pet_id=pet_id,
        period_from=period_from,
        period_to=period_to,
        timeline=rows,
        summary=summary,
        summary_by=summary_by,  # type: ignore[arg-type]
    )
