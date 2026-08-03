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


#: **LLM 없이 돈 노드 이름.** 프로세스 전역이고 하네스가 읽는다 (D-58 · `EngineUnavailable` 선례).
#:
#: 왜 남기나 — `generate_draft` 는 LLM 이 없으면 `draft = context` 로 폴백하고,
#: `verify_grounding` 은 draft 와 context 의 2-gram 을 비교한다. **폴백 경로에서는
#: 그 둘이 같으므로 판정이 항상 `근거있음`** 이다. 04 는 ④의 지표를
#: *"근거없음 탐지 재현율 — 놓치면 환각이 나간다"* 로 정했는데, 폴백에서는
#: 그 재현율이 **0인 채 100% 초록**으로 보인다.
#:
#: `deps.EngineUnavailable` 이 이미 같은 판단을 했다 —
#: *"조용히 스텁으로 내려가면 평가 지표가 스텁으로 산출된다. 그 지표는 오염된 것이므로
#: 기본은 실패다."* 같은 원칙을 LLM 폴백에도 적용한다. **끄지 않고 표시한다.**
LLM_FALLBACKS: set[str] = set()


def reset_llm_fallbacks() -> None:
    """측정 시작 전에 비운다. 하네스·테스트가 부른다."""
    LLM_FALLBACKS.clear()


#: 답변 초안 프롬프트. **5태스크 밖**이라 파인튜닝 대상이 아니다 (05 §4).
_DRAFT_PROMPT = (
    "주어진 근거만으로 보호자에게 답한다.\n"
    "**근거에 없는 수치·단위·종·물질을 만들지 않는다.**\n"
    "수치는 단위까지 근거 그대로 옮긴다. 위험도를 낮추는 완곡 표현을 쓰지 않는다."
)

#: LLM 트리아지 판정 프롬프트 (02 §6.2 · D-09).
#:
#: **이것이 없어서 `llm_level` 이 한 번도 안 세워졌다** (2026-08-02 흡수에서 확인).
#: `apply_gate` 는 `max(rule, llm)` 인데 `llm` 이 늘 `None` 이면
#: **하향 금지 게이트가 놀고 있는 것**이고, `overridden` 이 영원히 `False` 다 —
#: 산출물 ④에서 *"게이트가 작동했다"* 는 증거를 못 낸다.
_TRIAGE_PROMPT = (
    "근거를 읽고 긴급도를 **하나만** 고른다.\n"
    "EMERGENCY(지금 병원) · CALL_NOW(지금 전화) · VISIT_SOON(오늘 중 진료) · MONITOR(관찰)\n"
    "라벨만 출력한다. 판단이 서지 않으면 아무것도 출력하지 않는다 — "
    "**애매하면 비우는 것이 낮게 부르는 것보다 낫다** (D-13)."
)


def _call_raw(system: str, user_input: str, max_tokens: int) -> str | None:
    """5태스크 **밖**의 호출. 파인튜닝 태스크를 빌려 쓰지 않는다."""
    from ...models.serving.factory import get_client

    client = get_client()
    if client is None:
        LLM_FALLBACKS.add("(raw)")
        return None
    try:
        return client.run_raw(system, user_input, max_tokens=max_tokens).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("raw LLM 호출 실패: %s", type(e).__name__)
        LLM_FALLBACKS.add("(raw)")
        return None


def judge_triage(state: GraphState) -> GraphState:
    """④ 앞에서 **LLM 이 등급을 제안한다.** 코드가 허용목록으로 검증한다 (05 §4 ①과 같은 방식).

    목록 밖이거나 LLM 이 없으면 `llm_level` 을 **세우지 않는다** — 지어내지 않는다 (D-38).
    그러면 `apply_gate` 가 `rule_level` 만으로 판정하고, 그것이 정직한 결과다.
    """
    context = state.get("context", "")
    if not context:
        return {}  # type: ignore[return-value]
    raw = _call_raw(_TRIAGE_PROMPT, context, max_tokens=16)
    if not raw:
        return {}  # type: ignore[return-value]
    from ...triage.levels import TriageLevel

    for name in ("EMERGENCY", "CALL_NOW", "VISIT_SOON", "MONITOR"):
        if name in raw.upper():
            return {"llm_level": int(TriageLevel[name])}  # type: ignore[return-value]
    log.info("트리아지 라벨이 허용목록 밖이다: %r — 비워 둔다", raw[:40])
    return {}  # type: ignore[return-value]


def _call_llm(task: Task, user_input: str, max_tokens: int) -> str | None:
    """LLM 호출. **모델이 없거나** 실패하면 None. **폴백은 기록에 남긴다.**"""
    from ...models.serving.factory import get_client

    client = get_client()
    if client is None:
        LLM_FALLBACKS.add(task.value)
        return None

    try:
        return client.run(task, user_input, max_tokens=max_tokens).strip()
    except Exception as e:
        log.warning("%s LLM 호출 실패: %s", task.value, type(e).__name__)
        LLM_FALLBACKS.add(task.value)
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
    #
    # ⚠️ 400 이었을 때 실측(2026-08-03, ③ 학습데이터 생성 중 발견): 근거가
    # 4~5건 겹치는 질의(예: 발작·청소용품 노출)에서 응답이 문장 중간에
    # 잘렸다 — "5. 개에게 초콜" 처럼 단어 도중에 끊긴 사례가 489건 중
    # 200건 이상. 압축이 오히려 못다 한 말을 사실처럼 남기는 것이 원문을
    # 그대로 자르는 것보다 나쁘다.
    compressed = _call_llm(Task.COMPRESS, raw, max_tokens=700)
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
    # ⚠️ **태스크를 빌려 쓰지 않는다.** 흡수 전에는 `Task.COMPRESS` 를 불렀다 —
    # 초안 생성인데 압축 태스크다. 04 §3 은 태스크별 지표를 재는데, 한 태스크가
    # 두 일을 하면 **무엇을 잰 건지 모른다.**
    #
    # 05 §4 의 5태스크(①분류 ②슬롯 ③압축 ④검증 ⑤평이화)에 **"답변 생성"은 없다.**
    # 그래서 파인튜닝 대상이 아닌 **기본 모델 + 전용 프롬프트**로 부른다.
    # 6번째 태스크로 올릴지는 03·05 를 함께 고쳐야 하는 결정이라 남겨 둔다.
    draft = _call_raw(_DRAFT_PROMPT, user_input, max_tokens=300)

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
