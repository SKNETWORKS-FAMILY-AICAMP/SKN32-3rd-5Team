"""POST /api/records · GET /api/report — 다이어리 (02 §12).

스텁 단계다. 저장은 메모리이고 벡터DB 적재는 WS1이 붙인다.
계약만 먼저 고정해 프론트가 화면을 만들 수 있게 한다.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Query

from ..contracts import RecordCreate, RecordCreated, ReportResponse

router = APIRouter(prefix="/api", tags=["records"])

#: 스텁 저장소. WS1이 pgvector/Chroma 적재로 교체한다 (02 §11).
_RECORDS: dict[str, list[dict]] = {}


@router.post("/records", response_model=RecordCreated, status_code=201)
def create_record(rec: RecordCreate) -> RecordCreated:
    record_id = uuid.uuid4().hex[:12]
    payload = rec.model_dump()
    # 조류 전용 필드는 종이 맞을 때만 보관한다 (02 §12).
    if rec.species != "bird":
        payload.pop("droppings", None)
    payload["record_id"] = record_id
    _RECORDS.setdefault(rec.pet_id, []).append(payload)
    return RecordCreated(record_id=record_id, pet_id=rec.pet_id, indexed=False)


@router.get("/report", response_model=ReportResponse)
def report(
    pet_id: str = Query(max_length=64),
    period_from: str = Query(default=""),
    period_to: str = Query(default=""),
) -> ReportResponse:
    rows = _RECORDS.get(pet_id, [])
    return ReportResponse(
        pet_id=pet_id,
        period_from=period_from,
        period_to=period_to,
        timeline=rows,
        summary=f"기록 {len(rows)}건. 요약 생성은 구현 3단계에서 붙인다 (00 §8 로드맵 13번).",
    )
