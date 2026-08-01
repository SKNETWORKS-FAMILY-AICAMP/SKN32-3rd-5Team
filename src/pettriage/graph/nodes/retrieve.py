"""검색 노드 — **WS2 구현 대기.**

설계 근거: 02 §8 · D-10 · 05 §4

    **필터 구성은 전부 코드가 한다.** 여기에 LLM 을 넣지 않는다 (05 §4).
    유사도 임계 미만이면 **검색 실패로 보고 거절**한다 (02 §8.3).

⚠️ **임계값을 거절 장치로 믿지 말 것** (D-46).

    실측 결과 근거 있음(0.547~0.733)과 근거 없음(0.494~0.659)의 분포가 **겹친다.**
    `score_threshold=0.50` 은 완전히 무관한 것만 자르는 **최소 방어선**이고,
    도메인 밖은 ① 분류가, 근거 없음은 ④ 검증이 잡는다.

    **임계값을 올려서 정확도를 높이려 하지 말 것** — 근거가 있는 질의가 거절되고
    그것이 곧 과소평가다 (D-13).
"""

from __future__ import annotations

from typing import Any

from ..state import GraphState


def build_filter(state: GraphState) -> GraphState:
    """슬롯 → 검색 필터. 결정론이다.

    `species` 는 반드시 들어간다. 종이 `dog`·`cat` 이면 `mammal`·`all` 문서도
    함께 봐야 4단계가 성립한다 (D-39 — 고양이 자체 자료는 2단계뿐이다).

    목록은 **그대로 넘긴다** — `{"species": ["cat", "mammal", "all"]}`.
    저장소 문법으로의 번역은 `retrieval.to_chroma_where()` 가 한다.
    여기서 `$in` 을 쓰면 pgvector 로 옮길 때 이 노드를 고쳐야 한다.

    **`doc_type=recall` 은 사용자가 제품을 명시할 때만 넣는다** (D-46) —
    "사료 브랜드 추천해주세요" 가 리콜 목록의 제품명을 물어오는 일이 실측에서 나왔다.
    **리콜된 제품이 추천으로 읽히면 최악의 오답이다.**

    Returns:
        `{"where": {...}}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestFilter 참조")


def retrieve(state: GraphState, store: Any = None) -> GraphState:
    """벡터 검색 → 임계값 → **중복 접기.** 이 순서다.

    ```python
    hits = store.search(q, top_k=k, where=state["where"])
    hits = filter_by_threshold(hits, cfg.retrieval.score_threshold)
    hits = dedupe_by_substance(hits)          # ← 빠뜨리기 쉬운 단계
    ```

    **접기를 빠뜨리면 문맥이 같은 말로 채워진다.**

        `양파` 청크가 코퍼스에 8건이고, 고양이 질의는 D-39 병합으로 4건을 함께 본다.
        실측에서 상위 5가 `사람 음식 · 양파 · 양파 · 양파 · 토란` 으로 나왔다 —
        **`top_k=5` 가 실질 3종이다** (04 §2.5.6).

    접기는 **버리는 것이 아니다.** 흡수한 자료는 `Hit.merged_sources` 에 남고
    인용 화면은 `Hit.all_sources` 를 쓴다 — 근거가 하나뿐인 주장과
    넷이 같은 말을 하는 주장은 무게가 다르다 (02 §12).

    **접기를 임계값 뒤에 두는 이유** — 앞에 두면 임계 미달 청크가
    대표로 남아 통과할 수 있다. 자를 것을 먼저 자른다.

    Returns:
        `{"hits": [...]}`. 임계 통과분이 0건이면 부르는 쪽이
        `refused / 근거없음` 으로 보낸다 — **빈 결과는 실패가 아니라 신호다.**
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestRetrieve 참조")
