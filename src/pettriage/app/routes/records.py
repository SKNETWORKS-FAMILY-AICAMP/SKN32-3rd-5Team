"""POST /api/records · GET /api/report — 다이어리 (02 §12).

저장은 아직 메모리이고 벡터DB 적재는 WS1이 붙인다.
**요약은 2026-08-03 에 붙었다** — 05 §4 의 ③이 도는 곳이 여기다 (D-83).
집계는 코드가, 문장은 ③이 만든다 (`app/report.py`).

소유자 확인은 `deps.get_owner_id` 가 한다 — DB 구성에서는 Bearer 토큰 필수,
DB 없는 데모 구성에서는 단일 소유자다 (`records_store.py` 의 주의 참조).

⚠️ **요약은 기록 원문을 모델에 보낸다.** D-18(가상 프로필만 넣는다)이 지켜지는
   동안에만 안전하고, `privacy/` 필터가 붙기 전에는 실입력을 태우지 않는다 (D-36).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ..contracts import RecordCreate, RecordCreated, ReportResponse
from ..deps import get_owner_id, get_records
from ..records_store import RecordStore
from ..report import summarize_period

router = APIRouter(prefix="/api", tags=["records"])


@router.post("/records", response_model=RecordCreated, status_code=201)
def create_record(
    rec: RecordCreate,
    owner_id: str = Depends(get_owner_id),
    store: RecordStore = Depends(get_records),
) -> RecordCreated:
    record_id = store.add(owner_id, rec.model_dump())
    return RecordCreated(record_id=record_id, pet_id=rec.pet_id, indexed=False)


@router.get("/report", response_model=ReportResponse)
def report(
    pet_id: str = Query(max_length=64),
    period_from: str = Query(default=""),
    period_to: str = Query(default=""),
    owner_id: str = Depends(get_owner_id),
    store: RecordStore = Depends(get_records),
) -> ReportResponse:
    rows = store.timeline(owner_id, pet_id, period_from, period_to)
    summary, summary_by = summarize_period(rows, period_from, period_to)
    return ReportResponse(
        pet_id=pet_id,
        period_from=period_from,
        period_to=period_to,
        timeline=rows,
        summary=summary,
        summary_by=summary_by,  # type: ignore[arg-type]
    )
