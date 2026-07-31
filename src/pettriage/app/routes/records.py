"""POST /api/records · GET /api/report — 다이어리 (02 §12).

스텁 단계다. 저장은 메모리이고 벡터DB 적재는 WS1이 붙인다.
계약만 먼저 고정해 프론트가 화면을 만들 수 있게 한다.

⚠️ **인증이 없다** — `records_store.py` 의 주의 참조.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..contracts import RecordCreate, RecordCreated, ReportResponse
from ..deps import get_records
from ..records_store import RecordStore

router = APIRouter(prefix="/api", tags=["records"])


@router.post("/records", response_model=RecordCreated, status_code=201)
def create_record(rec: RecordCreate, store: RecordStore = Depends(get_records)) -> RecordCreated:
    record_id = store.add(rec.model_dump())
    return RecordCreated(record_id=record_id, pet_id=rec.pet_id, indexed=False)


@router.get("/report", response_model=ReportResponse)
def report(
    pet_id: str = Query(max_length=64),
    period_from: str = Query(default=""),
    period_to: str = Query(default=""),
    store: RecordStore = Depends(get_records),
) -> ReportResponse:
    rows = store.timeline(pet_id, period_from, period_to)
    return ReportResponse(
        pet_id=pet_id,
        period_from=period_from,
        period_to=period_to,
        timeline=rows,
        summary=f"기록 {len(rows)}건. 요약 생성은 구현 3단계에서 붙인다 (00 §8 로드맵 13번).",
    )
