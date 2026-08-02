"""GET /api/health · GET /api/triage-levels — 메타 정보.

`/api/triage-levels` 를 두는 이유: 프론트가 등급 이름·배지·문구를
자기 코드에 복사하면 D-39가 개정될 때 화면만 옛 표현으로 남는다.
**단일 출처 원칙**(00 §9.4)을 API로 강제한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ... import __version__
from ...config import get_config
from ...triage.levels import BIRD_FEEDING_LEVELS, EVIDENCE, FeedingLevel, TriageLevel
from ..contracts import DISCLAIMER, HealthResponse
from ..deps import get_engine
from ..engine import QAEngine

router = APIRouter(prefix="/api", tags=["meta"])


def _model_loaded(cfg) -> bool | None:
    """임베딩 모델이 올라와 있는가 (D-53).

    **모델을 로드하지 않는다.** 상태를 묻는 엔드포인트가 20초짜리 로딩을 일으키면
    헬스체크가 곧 장애가 된다. 이미 만들어진 임베더가 있으면 그 상태만 읽는다.

    `engine=stub` 은 벡터 검색을 하지 않으므로 **해당 없음(`None`)** 이다 —
    `False`(아직 안 올라옴)와 구분한다.
    """
    if cfg.serve.engine == "stub":
        return None
    try:
        from ...retrieval.embedder import BGEEmbedder, get_embedder

        emb = get_embedder(cfg.retrieval.embedding_model)
        return emb.loaded if isinstance(emb, BGEEmbedder) else None
    except Exception:  # noqa: BLE001 — 헬스체크가 예외로 죽으면 안 된다
        return None


@router.get("/health", response_model=HealthResponse)
def health(engine: QAEngine = Depends(get_engine)) -> HealthResponse:
    """실제 엔진과 **설정이 요구한 엔진**을 함께 돌려준다.

    둘이 다르면 폴백이 일어난 것이고, 그 상태로 산출한 평가 지표는 오염이다 (04 §8).
    화면과 스크립트가 이를 감지할 수 있어야 한다.
    """
    cfg = get_config()
    return HealthResponse(
        status="ok",
        engine=engine.name,
        engine_configured=cfg.serve.engine,
        profile=cfg_profile(),
        version=__version__,
        model_loaded=_model_loaded(cfg),
    )


def cfg_profile() -> str:
    import os

    return os.getenv("PETTRIAGE_PROFILE", "default")


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
