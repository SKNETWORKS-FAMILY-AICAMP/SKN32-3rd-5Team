"""③ 기간 요약 — **05 §4 의 ③이 실제로 도는 곳** (D-83).

설계 근거: D-02 · 03 §2 · 03 §4 · 05 §4 (③) · 02 §12

## 왜 여기인가

    D-02 는 다이어리 확장을 결정하면서 이렇게 적었다 —

        원안(중독 응급 QA)은 "무엇이 내부 문서인가" 에 답하지 못했고,
        **요약 태스크의 필연성도 약했다.**
        → 다이어리로 확장하면 **기간 요약 리포트라는 킬러 유스케이스**가 생긴다

    ③의 필연성은 처음부터 여기 있었다. 2026-08-03 까지 ③은 질의 그래프의
    `compress_context` 에서 돌았고, 이 자리는 문자열 하나로 비어 있었다 —

        summary=f"기록 {len(rows)}건. 요약 생성은 구현 3단계에서 붙인다"

    D-83 으로 둘을 맞바꿨다. 질의 경로에서 압축을 빼고 ③을 이 자리에 넣는다.

## 코드와 LLM 의 분업 (05 §4)

    **집계는 코드가 한다.** 건수·기간·증상 빈도·체중 변화는 결정론이고,
    같은 입력에 같은 답이 나와야 한다 (축① — 결정론은 코드로).
    LLM 은 그 집계를 **보호자가 읽을 문장으로 바꾼다.**

    그래서 모델이 없어도 리포트는 나온다. **다만 그 사실을 숨기지 않는다** —
    `ReportResponse.summary_by` 가 `"code"` 로 나간다 (04 §8).

## 🔴 진단하지 않는다

    기간 요약은 *"이 아이는 췌장염일 수 있습니다"* 를 말하는 자리가 아니다.
    관찰된 것과 변화만 적는다. 프롬프트(`models/prompts.py`)가 같은 말을 하고,
    여기서는 **입력 자체를 집계로 좁혀** 모델이 지어낼 여지를 줄인다.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

log = logging.getLogger(__name__)

#: 요약에 넣는 기록 상한. 넘으면 **최근 것부터** 남긴다.
#: 기간이 길면 입력이 길어지고, 길어지면 잘린다 — 잘리는 것보다 자른 것을 밝히는 편이 낫다.
MAX_ROWS = 60

#: 출력 상한. 🔴 **400 이 아니다.**
#: 2026-08-03 이서은 팀원 실측 — ③ 학습 데이터 489건 중 200건 넘게 `max_tokens=400`
#: 안에서 요약이 안 끝나 **단어 중간에서 끊겼다** (`"5. 개에게 초콜"`).
#: 기록 N일치는 검색 청크보다 길므로 이 자리에서 같은 일이 더 잘 난다.
_MAX_TOKENS = 700

#: 집계에서 세는 필드. 기록 스키마가 늘면 여기 추가한다.
_COUNTED = ("symptoms", "meals", "droppings")

#: 메모 한 건이 요약 입력에 실릴 수 있는 최대 글자수.
#: `RecordCreate.note` 는 4,000자까지 허용된다 — 60건이면 24만 자가 한 프롬프트로
#: 나가고 컨텍스트를 넘겨 호출이 통째로 실패한다. **상한은 건수가 아니라 길이다.**
_NOTE_CHARS = 300


def _digest(
    rows: list[dict[str, Any]], period_from: str, period_to: str, include_notes: bool = True
) -> str:
    """기록을 **결정론 집계**로 압축한다.

    이 문자열이 두 가지 일을 한다 —
      · LLM 에 주는 입력 (원문 전체를 주지 않는다)
      · 모델이 없을 때의 **폴백 요약 그 자체**

    둘을 같은 것으로 두는 이유는 04 §8 과 같다. 폴백 경로를 따로 만들면
    그 경로는 아무도 안 보고, 안 보는 경로는 조용히 썩는다.
    """
    if not rows:
        return f"{period_from or '전체'} ~ {period_to or '전체'} 기간에 기록이 없습니다."

    lines = [f"기간 {period_from or '전체'} ~ {period_to or '전체'} · 기록 {len(rows)}건"]

    dates = [str(r.get("recorded_at", ""))[:10] for r in rows if r.get("recorded_at")]
    if dates:
        lines.append(f"기록된 날짜 {dates[0]} ~ {dates[-1]} ({len(set(dates))}일)")

    for field in _COUNTED:
        counter: Counter[str] = Counter()
        for r in rows:
            v = r.get(field)
            if isinstance(v, list):
                counter.update(str(x) for x in v if x)
            elif v:
                counter.update([str(v)])
        if counter:
            top = " · ".join(f"{k} {n}회" for k, n in counter.most_common(8))
            lines.append(f"{field}: {top}")

    # 체중은 **처음과 마지막만** 본다. 중간값을 평균 내면 추세가 사라진다.
    weights = [(str(r.get("recorded_at", "")), r.get("weight_kg")) for r in rows]
    weights = [(d, w) for d, w in weights if isinstance(w, int | float)]
    if len(weights) >= 2:
        (d0, w0), (d1, w1) = weights[0], weights[-1]
        lines.append(f"체중 {w0}kg({d0[:10]}) → {w1}kg({d1[:10]}) · 차이 {w1 - w0:+.2f}kg")
    elif len(weights) == 1:
        lines.append(f"체중 {weights[0][1]}kg (1회 기록)")

    if include_notes:
        notes = [str(r.get("note", "")).strip()[:_NOTE_CHARS] for r in rows]
        notes = [n for n in notes if n]
        if notes:
            lines.append("메모:")
            lines += [f"  - {n}" for n in notes]
    elif any(str(r.get("note", "")).strip() for r in rows):
        lines.append("(메모는 요약 입력에서 제외했습니다 — 아래 §외부 전송 규칙)")

    return "\n".join(lines)


def _notes_may_leave_the_machine() -> bool:
    """**메모 원문을 외부 사업자에게 보내도 되는가.**

    🔴 지금은 언제나 거짓이다. `privacy/` 필터가 아직 없다 (D-36).

    이 함수가 있는 이유 — 라우트 머리말에 *"privacy 필터가 붙기 전에는 실입력을
    태우지 않는다"* 라고 **적어 두는 것만으로는 아무것도 막지 못한다.**
    D-40 이 반복해 말한 것이 그것이다: *지키기로 한 것이 아니라 못 어기게 만든다.*
    `records_store` 머리말의 예시(`'어제 구토, 주인 전화 010-1234-5678'`)가
    그대로 제3자 API 로 나가면 회수할 방법이 없다.

    필터가 생기면 여기 한 줄만 고친다.
    """
    return False


def _looks_truncated(text: str) -> bool:
    """**출력이 잘렸는가.**

    `LLMClient.run` 이 문자열만 돌려주어 `finish_reason` 을 볼 수 없다.
    그래서 끝 글자로 어림한다 — 문장 종결 부호나 종결 어미로 안 끝나면 의심한다.

    ⚠️ 어림이다. 제대로 하려면 `client.run` 이 `finish_reason` 을 함께 돌려줘야 한다.
       그 전까지는 **조용히 잘리는 것보다 시끄럽게 의심하는 편**이 낫다 —
       489건 중 200건이 잘리는 동안 아무도 몰랐던 것이 2026-08-03 의 교훈이다.
    """
    t = text.rstrip()
    return bool(t) and not t.endswith((".", "!", "?", "다", "요", "음", "함", ")", "」", "’"))


def summarize_period(
    rows: list[dict[str, Any]], period_from: str = "", period_to: str = ""
) -> tuple[str, str]:
    """기간 요약을 만든다.

    Returns:
        `(요약, 만든 주체)` — 주체는 `"model"` 또는 `"code"`.
        **폴백을 숨기지 않는다** (04 §8 · `HealthResponse.degraded` 와 같은 태도).
    """
    dropped = 0
    if len(rows) > MAX_ROWS:
        dropped = len(rows) - MAX_ROWS
        rows = rows[-MAX_ROWS:]  # 최근 것을 남긴다

    # ⚠️ 잘라낸 사실은 **화면에 나가는 요약**에 붙인다. digest 에만 붙이면
    #    LLM 이 성공했을 때 그 문장이 출력에 남는지가 **모델 재량**이 된다 —
    #    `timeline` 에는 전건이 실리므로 사용자는 어긋남을 알 방법이 없다.
    dropped_notice = (
        f"\n(기록 {dropped}건은 상한을 넘어 요약에서 제외했습니다. 기간 앞쪽입니다.)"
        if dropped
        else ""
    )

    include_notes = _notes_may_leave_the_machine()
    digest_for_user = _digest(rows, period_from, period_to, include_notes=True)
    digest_for_llm = _digest(rows, period_from, period_to, include_notes=include_notes)

    if not rows:
        return digest_for_user + dropped_notice, "code"

    # ⚠️ `graph/fallbacks.py` 의 `note_fallback` 을 **부르지 않는다.** 그 집합은
    #    질의 하나의 경계에서 `GraphEngine` 이 비우고 읽는다. 리포트는 그 경계 밖이라
    #    여기서 더해 봐야 아무도 읽지 않고, 다음 질의가 지운다.
    #    **폴백 기록은 `summary_by` 한 곳이다** (D-22 — 단일 출처).
    from ..models.serving.factory import get_client
    from ..models.tasks import Task

    client = get_client()
    if client is None:
        return digest_for_user + dropped_notice, "code"

    try:
        text = client.run(Task.COMPRESS, digest_for_llm, max_tokens=_MAX_TOKENS).strip()
    except Exception as e:  # noqa: BLE001
        log.warning("기간 요약 LLM 호출 실패: %s", type(e).__name__)
        return digest_for_user + dropped_notice, "code"

    if not text:
        return digest_for_user + dropped_notice, "code"

    if _looks_truncated(text):
        # **버리지 않는다.** 잘린 요약도 집계보다는 읽을 만하고, 버리면
        # 사용자는 폴백을 보면서 이유를 모른다. 대신 로그에 남기고 표시한다.
        log.warning(
            "기간 요약이 잘린 것으로 보인다 (max_tokens=%d) — 끝: %r", _MAX_TOKENS, text[-20:]
        )
        text += "\n(요약이 길어 끝까지 적지 못했을 수 있습니다. 아래 기록 원본을 함께 확인해 주세요.)"

    # 🔴 **연락처를 뺀다 (D-47).** 이 문장은 모델이 쓴 것이고, 모델은 검색 근거에
    #    없어도 자기 사전지식으로 `888-426-4435` 같은 미국 톨프리를 붙인다
    #    (`safety/contacts.py` 머리말). 질의 경로는 `SafetyEngine` 래퍼가 막지만
    #    **리포트는 그 관문 밖**이라 여기서 직접 훑는다.
    from ..safety import scrub_contacts

    scrubbed = scrub_contacts(text)
    if scrubbed.changed:
        log.warning("기간 요약에서 연락처 문장 %d개를 제거했다 (D-47)", len(scrubbed.removed))

    return scrubbed.text + dropped_notice, "model"
