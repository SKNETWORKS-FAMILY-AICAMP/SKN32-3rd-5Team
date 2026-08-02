"""검색 노드.

설계 근거: 02 §8 · D-10 · D-46 · 05 §4

    **필터 구성은 전부 코드가 한다.** 여기에 LLM 을 넣지 않는다 (05 §4).
    **결과가 0건이면 거절이다** — 점수가 낮은 것과 결과가 없는 것은 다르다 (02 §8.3).

⚠️ **임계값을 거절 장치로 믿지 말 것** (D-46).

    실측 결과 근거 있음(0.547~0.733)과 근거 없음(0.494~0.659)의 분포가 **겹친다.**
    `score_threshold=0.50` 은 완전히 무관한 것만 자르는 **최소 방어선**이고,
    도메인 밖은 ① 분류가, 근거 없음은 ④ 검증이 잡는다.

순서: 검색 → 임계값 → **중복 접기**. 접기를 뒤에 두는 이유는
임계 미달 청크가 대표로 남는 것을 막기 위함이다.
"""

from __future__ import annotations

import logging
from typing import Any

from ..state import GraphState

log = logging.getLogger(__name__)

#: 종별 필터 확장 규칙 (D-39).
#: 고양이 자체 자료가 2단계뿐이라 mammal·all 을 함께 봐야 4단계가 성립한다.
_SPECIES_FILTER: dict[str, list[str]] = {
    "dog":  ["dog", "mammal", "all"],
    "cat":  ["cat", "mammal", "all"],
    "bird": ["bird", "all"],   # 조류에 포유류 자료가 붙으면 안 된다 (D-10)
}


def build_filter(state: GraphState) -> GraphState:
    """슬롯 → 검색 필터. 결정론이다.

    `species` 는 반드시 들어간다. 종이 `dog`·`cat` 이면 `mammal`·`all` 문서도
    함께 봐야 4단계가 성립한다 (D-39 — 고양이 자체 자료는 2단계뿐이다).

    목록은 **그대로 넘긴다** — `{"species": ["cat", "mammal", "all"]}`.
    저장소 문법으로의 번역은 `retrieval.to_chroma_where()` 가 한다.

    Returns:
        `{"where": {...}}`
    """
    slots = state.get("slots") or {}
    species = slots.get("species")

    where: dict[str, Any] = {}

    if species in _SPECIES_FILTER:
        where["species"] = _SPECIES_FILTER[species]
    elif species:
        # 미지 종은 그대로 넣는다 — 확장 규칙이 없으니 추측하지 않는다.
        where["species"] = [species]

    return {"where": where}  # type: ignore[typeddict-item]


def retrieve(state: GraphState, store: Any = None) -> GraphState:
    """벡터 검색 → 임계값 → **중복 접기.** 이 순서다.

    접기는 **버리는 것이 아니다.** 흡수한 자료는 `Hit.merged_sources` 에 남고
    인용 화면은 `Hit.all_sources` 를 쓴다.

    Returns:
        `{"hits": [...]}`. 임계 통과분이 0건이면 부르는 쪽이
        `refused / 근거없음` 으로 보낸다 — **빈 결과는 실패가 아니라 신호다.**
    """
    query = state.get("question", "")
    where = state.get("where") or None

    # 설정값 로드
    try:
        from ...config import get_config
        cfg = get_config().retrieval
        top_k = cfg.top_k
        threshold = cfg.score_threshold
    except Exception:
        top_k = 5
        threshold = 0.50

    # 저장소가 주입되지 않으면 설정을 보고 만든다 (실서비스).
    if store is None:
        try:
            from ...retrieval import BGEEmbedder, ChromaStore
            store = ChromaStore(embedder=BGEEmbedder())
        except Exception as e:
            log.warning("기본 저장소 생성 실패 — 빈 결과 반환: %s", e)
            return {"hits": []}  # type: ignore[typeddict-item]

    from ...retrieval import dedupe_by_substance, filter_by_threshold

    hits = store.search(query, top_k=top_k, where=where)

    # 1) 임계값 미만 잘라내기 — 잡음 하한 (02 §8.3, D-46).
    filtered = filter_by_threshold(hits, threshold)

    # 2) 같은 물질 중복 접기 (D-46 후속).
    #    접기를 앞에 두면 임계 미달 청크가 대표로 남을 수 있어 반드시 뒤에 둔다.
    deduped = dedupe_by_substance(filtered)

    log.info(
        "retrieve: %d hits → %d ≥threshold → %d after dedupe",
        len(hits), len(filtered), len(deduped),
    )

    return {"hits": deduped}  # type: ignore[typeddict-item]