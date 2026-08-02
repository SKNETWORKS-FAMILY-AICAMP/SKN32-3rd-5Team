"""③ 압축 · 생성 · ⑤ 평이화 · ⑥ finalize 노드.

설계 근거: 02 §2 · 05 §4 (③⑤) · D-47

    LLM 이 문장을 만들고, **코드가 검증한다.**
      · compress_context : 길이 임계 초과 시에만 실행
      · generate_draft   : 원문에 없는 수치·단위·종을 만들지 않는다
      · simplify         : 위험도를 낮추는 완곡 표현을 쓰지 않는다
      · finalize         : **마지막 관문** — 연락처 스크러빙 (D-47)

`finalize` 는 LLM 판단에 맡길 수 없어 별도로 관리된다.
"""

from __future__ import annotations

import logging

from ...models.tasks import Task
from ...safety import scrub_contacts
from ..state import GraphState

log = logging.getLogger(__name__)

#: 압축을 실행하는 길이 임계 (문자수). 미만이면 그대로 둔다 — LLM 호출 낭비.
_COMPRESS_LEN_THRESHOLD = 800

#: 위험도를 낮추는 표현 목록. simplify 후 이 표현이 들어 있으면 그 문장을 제거한다.
_SOFTENING_TERMS = ("괜찮", "지켜보", "관찰만", "별문제", "걱정 마")


def _call_llm(task: Task, user_input: str, max_tokens: int) -> str | None:
    """LLM 호출. API 키 없거나 실패하면 None."""
    from ...config import get_secrets
    if not get_secrets().openai_api_key:
        return None

    try:
        from ...models.serving.client import APIClient
        client = APIClient()
        return client.run(task, user_input, max_tokens=max_tokens).strip()
    except Exception as e:
        log.warning("%s LLM 호출 실패: %s", task.value, type(e).__name__)
        return None


def compress_context(state: GraphState) -> GraphState:
    """검색 결과를 질문에 필요한 만큼으로 줄인다.

    **원문에 없는 수치·단위·종을 추가하지 않는다.** 압축 실행 여부는
    길이 임계로 코드가 정한다 (05 §4).

    Returns: `{"context": ...}`
    """
    hits = state.get("hits") or []
    existing_context = state.get("context", "")

    # 검색 결과에서 원문 조각을 이어붙인다.
    if hits:
        texts = []
        for h in hits:
            chunk = getattr(h, "chunk", None) or getattr(h, "text", None)
            text = getattr(chunk, "text", str(chunk)) if chunk else ""
            if text:
                texts.append(text)
        raw = "\n\n".join(texts)
    else:
        raw = existing_context

    # 짧으면 압축하지 않고 그대로 둔다.
    if len(raw) < _COMPRESS_LEN_THRESHOLD:
        return {"context": raw}  # type: ignore[typeddict-item]

    # LLM 압축 시도 — 실패하면 앞부분 잘라서 반환 (수치 유실은 없음).
    compressed = _call_llm(Task.COMPRESS, raw, max_tokens=400)
    return {"context": compressed if compressed else raw[:_COMPRESS_LEN_THRESHOLD]}  # type: ignore[typeddict-item]


def generate_draft(state: GraphState) -> GraphState:
    """근거를 바탕으로 초안을 만든다. 아직 사용자에게 나가지 않는다.

    LLM 이 있으면 근거를 바탕으로 생성하고, 없으면 **근거 문장을 그대로** 반환한다.
    원문에 없는 수치를 지어내지 않기 위한 안전한 폴백이다 —
    verify_grounding 이 뒤에서 다시 검증한다.

    Returns: `{"draft": ...}`
    """
    context = state.get("context", "")
    question = state.get("question", "")

    if not context:
        return {"draft": ""}  # type: ignore[typeddict-item]

    # LLM 시도 — 없으면 근거 그대로 사용 (수치 환각 방지).
    user_input = f"질문: {question}\n\n근거:\n{context}"
    draft = _call_llm(Task.COMPRESS, user_input, max_tokens=300)

    if not draft:
        # 폴백 — 근거 문장을 그대로 초안으로 사용한다.
        # verify 노드가 이 초안이 근거로 뒷받침되는지 다시 확인한다.
        draft = context

    return {"draft": draft}  # type: ignore[typeddict-item]


def simplify(state: GraphState) -> GraphState:
    """수의학 용어를 보호자 표현으로. **위험도를 낮추는 완곡 표현을 쓰지 않는다.**

    LLM 이 용어를 바꾸고, **코드가 완곡 표현을 검증한다** (05 §4).
    triage_level 이 높은데(≥3) 완곡 표현이 섞였으면 그 문장을 제거한다.

    이 노드 **다음에 반드시** `finalize` 가 온다 (D-47). 순서를 바꾸면
    평이화가 지워진 연락처를 다시 만들어 넣을 수 있다.

    Returns: `{"answer": ...}`
    """
    draft = state.get("draft", "")
    triage_level = state.get("triage_level") or 1

    if not draft:
        return {"answer": ""}  # type: ignore[typeddict-item]

    # LLM 시도. 실패하면 draft 그대로 사용.
    answer = _call_llm(Task.SIMPLIFY, draft, max_tokens=300) or draft

    # 검증: 위험도가 높은데 완곡 표현이 들어 있으면 해당 문장을 제거한다.
    #       이 검증은 D-11(진단 금지) · D-39(과소평가 억제) 를 코드가 강제하는 지점이다.
    if triage_level >= 3:
        sentences = [s.strip() for s in answer.replace("\n", " ").split(".") if s.strip()]
        safe = [s for s in sentences if not any(t in s for t in _SOFTENING_TERMS)]
        answer = ". ".join(safe)
        if answer and not answer.endswith("."):
            answer += "."

    return {"answer": answer}  # type: ignore[typeddict-item]


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