"""③ 압축 · 생성 · ⑤ 평이화 노드 — **WS2 구현 대기.**

설계 근거: 02 §2 · 05 §4 (③⑤) · D-47

`finalize` 만 **지금 구현되어 있다.** 연락처 차단은 LLM 판단에 맡길 수 없어서다.
"""

from __future__ import annotations

from ...safety import scrub_contacts
from ..state import GraphState


def compress_context(state: GraphState) -> GraphState:
    """검색 결과를 질문에 필요한 만큼으로 줄인다.

    **원문에 없는 수치·단위·종을 추가하지 않는다.** 압축 실행 여부는
    길이 임계로 코드가 정한다 (05 §4).

    Returns: `{"context": ...}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestGenerate 참조")


def generate_draft(state: GraphState) -> GraphState:
    """근거를 바탕으로 초안을 만든다. 아직 사용자에게 나가지 않는다.

    Returns: `{"draft": ...}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestGenerate 참조")


def simplify(state: GraphState) -> GraphState:
    """수의학 용어를 보호자 표현으로. **위험도를 낮추는 완곡 표현을 쓰지 않는다.**

    용어집 준수는 코드가 검증한다 (05 §4).

    **문체는 존댓말이다.** 코퍼스 청크는 검색 대상이라 평서체(`~다`)로 문장화하지만
    (`ingest/templates.py`), 화면에 뜨는 답변은 보호자가 읽는 말이다.
    응급 상황에서 명령조로 말하지 않는다 — `safety.GUIDANCE` 가 기준 문체다.

    이 노드 **다음에 반드시** `finalize` 가 온다. 순서를 바꾸면
    평이화가 지워진 연락처를 다시 만들어 넣을 수 있다.

    Returns: `{"answer": ...}`
    """
    raise NotImplementedError("WS2: tests/todo/test_graph_nodes.py::TestGenerate 참조")


def finalize(state: GraphState) -> GraphState:
    """**마지막 관문** — 사용자에게 나가기 직전에 연락처를 뺀다 (D-47).

    코퍼스 응급 자료가 전부 미국 것이라 답변에 미국 톨프리 번호가 실릴 수 있다.
    **국내 사용자가 그 번호로 걸면 아무 일도 일어나지 않는다** —
    응급 상황에서 걸리지 않는 번호는 오답보다 나쁘다.

    ④ 검증이 `근거없음` 으로 잡아주기를 기대하지 않는다. **판정은 보장이 아니다.**

    거절·되묻기 응답도 함께 통과시킨다 — 거절 문구에 연락처를 넣는 실수를 막는다.

    Returns: `{"answer": ..., "removed_contacts": [...]}`
    """
    answer = state.get("answer") or ""
    if not answer:
        return {}
    r = scrub_contacts(answer)
    if not r.changed:
        return {}
    return {"answer": r.text, "removed_contacts": r.removed}
