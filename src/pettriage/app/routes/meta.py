"""GET /api/health · GET /api/triage-levels — 메타 정보.

`/api/triage-levels` 를 두는 이유: 프론트가 등급 이름·배지·문구를
자기 코드에 복사하면 D-39가 개정될 때 화면만 옛 표현으로 남는다.
**단일 출처 원칙**(00 §9.4)을 API로 강제한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import __version__
from ...triage.levels import BIRD_FEEDING_LEVELS, EVIDENCE, FeedingLevel, TriageLevel
from ..contracts import DISCLAIMER, HealthResponse
from ..deps import get_engine
from ..engine import QAEngine

router = APIRouter(prefix="/api", tags=["meta"])


@router.get("/health", response_model=HealthResponse)
def health(engine: QAEngine = Depends(get_engine)) -> HealthResponse:
    return HealthResponse(status="ok", engine=engine.name, version=__version__)


@router.get("/triage-levels")
def triage_levels() -> dict:
    """등급 정의 + **코퍼스 근거**를 함께 내려보낸다 (D-39).

    발표·시연에서 "이 등급 이름은 어디서 왔나"에 화면에서 바로 답할 수 있다.
    """
    return {
        "disclaimer": DISCLAIMER,
        "levels": [
            {
                "level": int(lv),
                "name": lv.name,
                "badge": lv.badge,
                "message": lv.message,
                "evidence": {"source_id": EVIDENCE[lv][0], "quote": EVIDENCE[lv][1]},
            }
            # 높은 등급이 위에 오도록 내림차순
            for lv in sorted(TriageLevel, reverse=True)
        ],
        "feeding_levels": [
            {"level": int(fl), "name": fl.name, "label": fl.label} for fl in FeedingLevel
        ],
        # 조류는 SAFE를 노출하지 않는다 — 출처 간 티어가 충돌한다 (D-39).
        "bird_feeding_levels": sorted(int(fl) for fl in BIRD_FEEDING_LEVELS),
    }
