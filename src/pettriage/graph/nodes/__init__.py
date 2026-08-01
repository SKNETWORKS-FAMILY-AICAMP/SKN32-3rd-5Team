"""그래프 노드 — **WS2가 채운다.**

설계 근거: docs/02_시스템-아키텍처.md §2 · §6 · §7 · docs/05 §4

각 노드는 `GraphState` 를 받아 **바뀐 키만** 돌려준다.
아래 함수들은 서명과 계약만 정의되어 있고 본문이 비어 있다.
`tests/todo/test_graph_nodes.py` 가 각 노드가 만족해야 할 조건을 담고 있으니,
**그 테스트를 초록으로 만드는 것이 이 작업의 완료 기준이다.**

```bash
pytest -m todo          # 남은 일 목록
pytest -m todo -k slot  # 한 노드만
```

## 노드 순서 (02 §2)

```
classify_intent → extract_slots ─┬─ 결측 → ask_clarify (상한 2회)
                                 └─ 충족 → build_filter → retrieve
                                            → compute → compress
                                            → generate → triage
                                            → verify_grounding ─┬─ 통과 → simplify
                                                                └─ 실패 → retrieve (1회)
                                                                          simplify → finalize
```

## 절대 어기면 안 되는 것

- **필터는 코드가 만든다.** `build_filter` 에 LLM을 넣지 않는다 (05 §4)
- **종이 없으면 검색하지 않는다** (D-10). `extract_slots` 에서 막는다
- **유사도 임계 미만은 거절**이다 (02 §8.3). 낮은 점수 문서로 답을 만들지 않는다
- **트리아지는 `apply_gate` 를 거친다** (D-09). `max()` 를 직접 쓰지 않는다
- **`finalize` 가 마지막이다** (D-47). 이 뒤에 문장을 덧붙이면 연락처 차단이 무력해진다
- **사용자에게 나가는 문장은 존댓말이다.** 청크의 평서체(`~다`)를 그대로 내보내지 않는다
"""

from .classify import classify_intent
from .compute import compute_metrics
from .generate import compress_context, finalize, generate_draft, simplify
from .retrieve import build_filter, retrieve
from .slots import ask_clarify, extract_slots
from .triage import decide_triage
from .verify import verify_grounding

#: 노드 구현이 끝나면 **WS2 가 True 로 바꾼다.**
#: 이 값이 False 인 동안 `GraphEngine` 은 생성 자체가 실패한다 —
#: 반쯤 구현된 그래프로 평가를 돌리면 지표가 오염되기 때문이다 (04 §8).
NODES_IMPLEMENTED = False

__all__ = [
    "NODES_IMPLEMENTED",
    "ask_clarify",
    "build_filter",
    "classify_intent",
    "compress_context",
    "compute_metrics",
    "decide_triage",
    "extract_slots",
    "finalize",
    "generate_draft",
    "retrieve",
    "simplify",
    "verify_grounding",
]
