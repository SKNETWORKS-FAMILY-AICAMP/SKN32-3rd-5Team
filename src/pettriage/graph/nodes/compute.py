"""계산 노드 (비-RAG) — **WS2 구현 대기.**

설계 근거: 02 §7 · D-16 · 05 §4

    **수치는 벡터 검색으로 찾지 않는다** (D-16). 관계형 테이블을 조회하고
    계산은 코드가 한다. LLM 은 여기 관여하지 않는다.
"""

from __future__ import annotations

from ..state import GraphState


def compute_metrics(state: GraphState) -> GraphState:
    """체중당 섭취량·권장 열량 등을 계산한다.

    단위는 내부적으로 `weight_g` 하나로 통일한다 (D-17).
    kg↔g 혼용이 곧 용량 계산 오류다.

    **정량으로 답할 근거가 없으면 여기서 판정한다** (D-46).

        검색은 정성 근거를 잘 물어온다 — "앵무새 체중 100g 기준 초콜릿 몇 g부터"
        라는 질의가 조류 초콜릿 청크를 0.659로 가져온다. 검색은 맞다. 그 자료는 실재한다.
        **틀리는 것은 정량 질문에 정량 근거가 없다는 판정**이고, 그건 검색이 아니라 여기 일이다.

        규칙 테이블에 해당 (물질 × 종) 행이 없으면 `computed` 를 비우고
        부르는 쪽이 **정성 답변으로 내려가거나 `refused / 판정불가`** 로 보낸다.
        **조류는 체중당 임계치가 코퍼스에 0건이다** (D-09 개정) — 수치를 만들어내면 그게 환각이다.

    **바닥 등급까지 여기서 만든다** (D-50).

    ```python
    mg = to_mg_per_kg(amount_g * 1000 / weight_kg, "mg/kg")
    verdict = rule_level_for(substance, species, mg)
    ```

    `verdict.level` 이 `None` 인 경우가 둘이고, **둘 다 실패가 아니다.**

      1. 계산 가능한 역치가 없다 — **조류가 전부 여기다** (D-09)
      2. 출처 간 수치가 10배 이상 벌어졌다 — 어느 쪽이 맞는지 모른다 (건포도 1,000배 사례)

    부르는 쪽은 **정성 답변으로 내려가거나** `refused / 판정불가` 로 보낸다 (D-46).
    `verdict.reason` 을 로그에 남긴다 — 왜 정량을 포기했는지가 오류 분석의 입력이다 (04 §7).

    Returns:
        `{"computed": {"dose_per_kg": ..., "unit": ...}, "rule_level": ...}`.
        근거가 없으면 **빈 dict** — 지어낸 수치를 넣지 않는다.
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestCompute 참조")
