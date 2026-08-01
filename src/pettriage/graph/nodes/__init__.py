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

## 상수도 여기서 내보낸다

`ALLOWED_INTENTS`(①의 허용목록)·`MAX_RETRY`(④의 재검색 상한)는 각 노드 파일에 있지만
**`tests/todo` 가 이 패키지에서 임포트한다.** 여기 안 실으면 테스트가 본문에 닿기도 전에
`ImportError` 로 죽어, 착수하는 사람이 **자기 코드 문제로 오해한다** (2026-08-02 발견).

## 노드 순서 (02 §2)

```
classify_intent ─┬─ general → refuse (범위밖 · D-46) ────────────────┐
                 │                                                   │
                 └─ 그 외 → extract_slots ─┬─ 결측·물질미상 → ask_clarify (상한 2회 · D-49)
                                           │                         │
                                           └─ 충족 → build_filter → retrieve
                                                       → compute → compress
                                                       → generate → decide_triage
                                                       → verify_grounding
                                                            ├─ 통과 → simplify ──┐
                                                            └─ 실패 → retrieve (1회)
                                                                     → 그래도 실패 → refuse
                                                                                   │
                                     모든 경로가 마지막에 만난다 ──────────────► finalize
```

**`finalize` 는 `answered` 만 통과시키는 것이 아니다.** 거절·되묻기도 지나간다 —
거절 문구에 미국 핫라인을 넣는 실수를 막기 위해서다 (D-47).

## 절대 어기면 안 되는 것

- **필터는 코드가 만든다.** `build_filter` 에 LLM을 넣지 않는다 (05 §4)
- **종이 없으면 검색하지 않는다** (D-10). `extract_slots` 에서 막는다
- **⚠️ 유사도 점수로 거절 분기를 만들지 않는다 (D-46).** 아래 별도 항목 참조
- **증상만 있거나 물질이 미상이면 되묻는다 (D-49).** 검색으로 원인을 찾지 않는다
- **트리아지는 `apply_gate` 를 거친다** (D-09). `max()` 를 직접 쓰지 않는다.
  규칙이 낸 값은 확정이 아니라 **바닥**이므로, 적중했다고 LLM 판정을 건너뛰지 않는다 (D-50)
- **`finalize` 가 마지막이다** (D-47). 이 뒤에 문장을 덧붙이면 연락처 차단이 무력해진다
- **사용자에게 나가는 문장은 존댓말이다.** 청크의 평서체(`~다`)를 그대로 내보내지 않는다

## 속도 — 캐시보다 먼저 지킬 것 셋 (D-53)

**지연은 LLM 호출 6번이 지배한다.** 질의 임베딩은 실측 **193ms** 로 전체의 1% 남짓이다.
그런데 아래 셋을 안 지키면 **LLM 과 무관한 곳에서 수십 초를 잃는다.**

- **모델을 프로세스에 상주시킨다.** ✅ `get_embedder` 에 `lru_cache` 가 붙어 있으니
  **그냥 팩토리를 쓰면 된다.** 직접 `BGEEmbedder(...)` 를 만들지 않는다 —
  그러면 인스턴스마다 따로 로드한다.
  로드 시간은 **아직 측정된 적이 없다.** 처음 도는 사람이 로그(`임베딩 모델 로드 완료 — …s`)를
  보고 04 §8 재현성 표에 적는다
- **정량 임계치 테이블은 `lru_cache`.** 12행짜리를 질의마다 재파싱하지 않는다
- **프롬프트 프리픽스를 고정한다.** `models/prompts.py` 가 이미 태스크별 고정 문자열이라
  KV 프리픽스 캐시가 그대로 먹힌다. 문자열을 동적으로 조립하면 그 이점이 사라진다

**워밍업은 두 곳에서 이미 돈다** — FastAPI `lifespan`(서버 기동 시)과
평가 하네스(측정 전 1회 버림). 노드에서 따로 부를 필요가 없다.

**그 밖의 캐시는 붙이지 않는다.** 05 §2 가 조각 11(캐싱)을 🔶 선택으로 판정했고,
**지연을 아직 재 보지 않았다.** 하네스에 측정이 들어가 있으니(04 §4.3) 숫자를 먼저 본다.

> 🔴 **캐시 금지 — 안전 장치를 건너뛰는 지름길이 되면 안 된다** (D-53)
> **프로필**(D-52 · 낡은 값이 더 낡는다) · **트리아지 판정**(게이트를 건너뛴다) ·
> **`finalize` 이전 출력**(히트마다 연락처가 나간다).

## ⚠️ 유사도 임계값은 거절을 만들지 못한다 (D-46)

**이 문단은 2026-08-02에 정반대로 고쳐졌다.** 이전 판은 여기에
*"유사도 임계 미만은 거절이다 — 낮은 점수 문서로 답을 만들지 않는다"* 라고 적혀 있었다.
**실측이 그것을 뒤집었다.**

```
근거 있는 질의   0.547 ~ 0.733
근거 없는 질의   0.494 ~ 0.659      ← 겹친다
```

어떤 값을 잘라도 한쪽이 틀린다. 0.66으로 올리면 근거 있는 질의를 대량 거절하고,
0.49로 내리면 도메인 밖 질의가 통과한다. 실제로 *"고양이 이름 지어주세요"* 가 **0.550**,
*"캣타워 추천"* 이 **0.574** 로 양성 최저점(PTFE 0.547)보다 높다.

그래서 **`score_threshold: 0.50` 은 명백한 잡음만 걷어내는 하한**이고,
거절은 **앞의 `classify_intent`(범위밖)** 와 **뒤의 `verify_grounding`(근거없음)** 이 만든다.

```python
# ✗ 하면 안 되는 것 — 근거 있는 질의를 대량으로 거절한다
if hits[0].score < threshold:
    return {"status": "refused", "refusal_reason": "근거없음"}

# ✓ retrieve 가 할 일은 세 줄뿐이다
hits = store.search(query, where=state["where"], top_k=cfg.top_k)
hits = filter_by_threshold(hits, cfg.score_threshold)   # 잡음 하한
hits = dedupe_by_substance(hits)                        # 같은 물질 접기
```

**결과가 0건이면 거절이 맞다.** 점수가 낮은 것과 결과가 없는 것은 다르다.

재현: `python scripts/calibrate_threshold.py` · 근거: 02 §8.3 · 04 §2.5.4
"""

from .classify import ALLOWED_INTENTS, classify_intent
from .compute import compute_metrics
from .generate import compress_context, finalize, generate_draft, simplify
from .retrieve import build_filter, retrieve
from .slots import ask_clarify, extract_slots
from .triage import decide_triage
from .verify import MAX_RETRY, verify_grounding

#: 노드 구현이 끝나면 **WS2 가 True 로 바꾼다.**
#: 이 값이 False 인 동안 `GraphEngine` 은 생성 자체가 실패한다 —
#: 반쯤 구현된 그래프로 평가를 돌리면 지표가 오염되기 때문이다 (04 §8).
NODES_IMPLEMENTED = False

__all__ = [
    "ALLOWED_INTENTS",
    "MAX_RETRY",
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
