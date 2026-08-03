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
from functools import lru_cache
from typing import Any

from ..state import GraphState

log = logging.getLogger(__name__)

#: 과다 인출 배수. **접기가 자리를 비우므로 미리 더 가져온다.**
#: 3배로 둔 근거 — 코퍼스가 물질 단위 청크(D-14)라 같은 물질이 여러 출처에서 오고,
#: 실측에서 5건 인출이 접은 뒤 3건으로 줄었다(≈60% 생존). 3배면 `top_k` 를 채운다.
_OVERFETCH = 3

#: 종별 필터 확장 규칙 (D-39).
#: 고양이 자체 자료가 2단계뿐이라 mammal·all 을 함께 봐야 4단계가 성립한다.
_SPECIES_FILTER: dict[str, list[str]] = {
    "dog": ["dog", "mammal", "all"],
    "cat": ["cat", "mammal", "all"],
    "bird": ["bird", "all"],  # 조류에 포유류 자료가 붙으면 안 된다 (D-10)
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


def _augment(state: GraphState) -> str:
    """질의에 **정규화된 물질명**을 덧붙인다 (D-62).

    ⚠️ 흡수 전에는 `state["question"]` 원문만 임베딩했다. 그러면
    **별칭 표가 검색에 아무 영향을 못 준다** —

        "5kg 고양이가 대파를 40g쯤 뜯어 먹었어요"
          코퍼스에 `대파` 는 0건.  `알리움류(양파·마늘·리크·차이브)` 로 적혀 있다
          → 원문만 넣으면 못 찾는다.  골든셋 `G-039` 가 실패하던 바로 그 이유다

    ②가 `대파 → 알리움류` 로 올려 뒀으므로 그 이름을 검색어에 싣는다.
    하나로 못 좁힌 경우(`모호`)는 **후보를 전부** 싣는다 — 하나를 고르면
    나머지를 배제한 것이고 그 배제가 곧 진단이다 (D-11 · D-49 · D-58).

    원문을 **지우지 않고 덧붙인다.** 원문의 증상·정황 서술이 검색 신호이고,
    확장어가 그것을 밀어내면 D-59 ②가 든 세 번째 실패(*"확장어가 검색을 엉뚱한
    쪽으로 끌고 간다"*)가 된다.
    """
    question = state.get("question", "")
    extra: list[str] = []

    substance = (state.get("slots") or {}).get("substance")
    if substance and substance not in question:
        extra.append(substance)

    for cand in state.get("substance_candidates") or []:  # type: ignore[typeddict-item]
        if cand not in question and cand not in extra:
            extra.append(cand)

    if not extra:
        return question
    log.info("retrieve: 검색어 보강 %s", extra)
    return f"{question} {' '.join(extra)}"


@lru_cache(maxsize=1)
def _default_store() -> Any:
    """주입이 없을 때 쓰는 저장소. **프로세스에 하나만 만든다** (D-53).

    ⚠️ 예전에는 `retrieve` 안에서 **매 호출마다** 이렇게 만들었다 —

        store = ChromaStore(embedder=BGEEmbedder())

    `embedder.get_embedder` 의 docstring 이 바로 이 코드를 **하면 안 되는 예**로
    적어 두고 있었다: *"✗ 노드 안에서 이렇게 쓰면 — 그리고 이게 자연스러운 코드다.
    인스턴스가 매번 새로 생기면 그 캐시가 아무 소용이 없다."*
    `BGEEmbedder()` 를 직접 만들면 **lru_cache 를 우회해 질의마다 모델을 다시 올린다.**

    2026-08-02 첫 실측에서 그대로 드러났다 — **LLM 을 한 번도 안 부른 조건**에서
    p50 이 4.71초였다. 임베딩 자체는 실측 193ms 다. 나머지는 재로딩이다.
    하네스가 워밍업 1회를 버리는데, 매번 다시 올리면 그 워밍업도 소용이 없다.

    설정(`retrieval.persist_dir`·`collection`)도 이제 실제로 읽는다 — 예전에는
    `ChromaStore` 기본값(`.chroma` · `external`)이 우연히 같아서 안 드러났다.
    다르게 두는 순간 **설정과 다른 컬렉션을 조용히 뒤졌을 것이다** (D-40).

    Returns:
        `VectorStore` 또는 **`None`** — 만들지 못하면 부르는 쪽이 빈 결과로 간다.
        여기서 터지면 질의 하나가 아니라 서비스가 죽는다.
    """
    try:
        from ...config import get_config
        from ...retrieval import ChromaStore, get_embedder

        cfg = get_config().retrieval
        return ChromaStore(
            embedder=get_embedder(cfg.embedding_model),  # ← 팩토리를 쓴다 (D-53)
            persist_dir=cfg.persist_dir,
            collection=cfg.collection,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("기본 저장소 생성 실패 — 빈 결과 반환: %s", e)
        return None


def reset_default_store() -> None:
    """설정을 바꿔 가며 도는 하네스·테스트가 부른다."""
    _default_store.cache_clear()


def retrieve(state: GraphState, store: Any = None) -> GraphState:
    """벡터 검색 → 임계값 → **중복 접기.** 이 순서다.

    접기는 **버리는 것이 아니다.** 흡수한 자료는 `Hit.merged_sources` 에 남고
    인용 화면은 `Hit.all_sources` 를 쓴다.

    Returns:
        `{"hits": [...]}`. 임계 통과분이 0건이면 부르는 쪽이
        `refused / 근거없음` 으로 보낸다 — **빈 결과는 실패가 아니라 신호다.**
    """
    query = _augment(state)
    where = state.get("where") or None

    # 설정값 로드
    try:
        from ...config import get_config

        cfg = get_config().retrieval
        top_k = cfg.top_k
        threshold = cfg.score_threshold
    except Exception as e:  # noqa: BLE001
        # ⚠️ **조용히 넘어가지 않는다.** `ConfigNotFound` 는 *"평가 프로파일이 무시된 채
        #    지표가 산출되는 것"* 을 막으려고 만든 예외인데, 여기서 말없이 삼키면
        #    그 예외가 존재할 이유가 없어진다 (`config.py` 머리말 · D-69).
        log.warning("검색 설정을 읽지 못해 기본값으로 간다 — %s: %s", type(e).__name__, e)
        top_k = 5
        threshold = 0.50

    # 저장소가 주입되지 않으면 설정을 보고 만든다 (실서비스).
    if store is None:
        store = _default_store()
        if store is None:
            return {"hits": []}  # type: ignore[typeddict-item]

    from ...retrieval import dedupe_by_substance, filter_by_threshold

    # 🔴 **접은 뒤에 자른다** — 그 전에는 `top_k` 가 실현된 적이 없다.
    #
    #    예전 순서는 `search(top_k=5) → threshold → dedupe` 였다. 접기는 중복을
    #    흡수할 뿐 **빈자리를 채우지 않으므로** `top_k=5` 로 설정한 실행이 실제로는
    #    근거 3건으로 답했다 (`dedupe_by_substance` docstring 이 그 현상을 그대로
    #    적어 두었다). 그런데 리포트 provenance 에는 `top_k: 5` 가 기록된다 —
    #    **실제보다 큰 값을 적는 것**이라 04 §8 재현성에 걸린다.
    #
    #    과다 인출한 뒤 접고, 마지막에 `top_k` 로 자른다.
    retried = state.get("retry_count", 0) > 0
    fetch_k = top_k * (_OVERFETCH * 2 if retried else _OVERFETCH)

    # 재검색은 **다른 것을 가져와야 의미가 있다.** 예전에는 `_retry` 가 카운터만 올리고
    # 쿼리·필터·`top_k` 가 모두 같아 **글자 그대로 같은 히트**를 돌려줬다 — 재검색이
    # 근거를 새로 붙일 수단이 없었다. 넓히는 방향은 잡음 하한을 내리는 쪽이다.
    # D-46 이 *"임계값으로는 거절을 만들 수 없다"* 로 이미 결론냈으므로 안전하다.
    if retried:
        threshold = 0.0

    hits = store.search(query, top_k=fetch_k, where=where)

    # 1) 임계값 미만 잘라내기 — 잡음 하한 (02 §8.3, D-46).
    filtered = filter_by_threshold(hits, threshold)

    # 2) 같은 물질 중복 접기 (D-46 후속).
    #    접기를 앞에 두면 임계 미달 청크가 대표로 남을 수 있어 반드시 뒤에 둔다.
    deduped = dedupe_by_substance(filtered)[:top_k]

    log.info(
        "retrieve: fetch_k=%d(top_k=%d%s) → %d hits → %d ≥%.2f → %d after dedupe/cut",
        fetch_k,
        top_k,
        " · 재검색" if retried else "",
        len(hits),
        len(filtered),
        threshold,
        len(deduped),
    )

    return {"hits": deduped}  # type: ignore[typeddict-item]
