"""`SafetyEngine` — 어떤 엔진을 꽂아도 D-47 이 지켜지게 하는 래퍼.

설계 근거: docs/06_설계결정기록.md · D-47 · D-40

왜 래퍼인가
----------
2026-08-02 검토에서 드러난 것 —

    `scrub_contacts` 의 **유일한 호출부가 `graph/nodes/generate.py::finalize`** 였다.
    그런데 `finalize` 는 `GraphEngine` 전용이고, `GraphEngine` 은
    `NODES_IMPLEMENTED=False` 라 생성조차 되지 않는다.
    **즉 그때까지 `/api/ask` 응답은 어떤 연락처 필터도 통과하지 않았다.**

차단이 "응답을 만드는 모든 경로가 각자 불러야 하는 함수"로 있으면
**새 엔진을 꽂는 사람이 모르면 그대로 뚫린다.** D-40 이 계약·주입을 단일 출처로 둔 이유가
이것이다 — *지키기로 한 것이 아니라 못 어기는 것.*

그래서 `deps.get_engine()` 이 **무엇을 만들든 이 래퍼로 감싼다.**
`QAEngine` 프로토콜을 그대로 만족하므로 라우터·계약·테스트는 손대지 않는다.

무엇을 거르나 — `full_text` 에 들어가는 것 전부
--------------------------------------------
`AskResponse.full_text` 는 `answer` 뒤에 `escalation_conditions` 와 `disclaimer` 를
이어 붙인다. `answer` 만 걸러서는 **뒤에 붙는 조건에서 번호가 되살아난다.**
같은 검토에서 실측으로 확인했다.

    answer   : 초콜릿은 개에게 독성이 있습니다. 연락처는 안내하지 않습니다.
    full_text: … 다음 증상이 나타나면 즉시 알리세요 — 구토가 계속되면 855-764-7661 로 연락. …

따라서 `answer` · `clarify.question` · `refusal.message` · `refusal.advice` ·
`triage.message` · `triage.escalation_conditions` 를 **전부** 거른다.

`escalation_conditions` 만 항목 단위인 이유
----------------------------------------
코퍼스에 실제 유출이 하나 있다 (`F-021-012`). 문장 단위로 처리하면 그 리스트의
나머지 네 조건("원인 물질에서 떼어놓기" 등)까지 함께 사라진다.
`safety.scrub_items` 의 docstring 참조.
"""

from __future__ import annotations

import logging

from ..safety import GUIDANCE, scrub_contacts, scrub_items
from .contracts import AskRequest, AskResponse
from .engine import QAEngine
from .session import Session

log = logging.getLogger(__name__)


def _scrub_text(value: str | None) -> tuple[str | None, list[str]]:
    if not value:
        return value, []
    r = scrub_contacts(value)
    return (r.text, r.removed) if r.changed else (value, [])


def scrub_response(resp: AskResponse) -> AskResponse:
    """응답 1건에서 국내에서 쓸 수 없는 연락처를 뺀다.

    바뀐 것이 없으면 **원본 객체를 그대로 돌려준다** — 불필요한 재검증을 피한다.
    """
    removed: list[str] = []
    changes: dict[str, object] = {}

    answer, got = _scrub_text(resp.answer)
    removed += got
    if got:
        changes["answer"] = answer

    if resp.clarify is not None:
        q, got = _scrub_text(resp.clarify.question)
        removed += got
        if got:
            changes["clarify"] = resp.clarify.model_copy(update={"question": q})

    if resp.refusal is not None:
        msg, g1 = _scrub_text(resp.refusal.message)
        adv, g2 = _scrub_text(resp.refusal.advice)
        removed += g1 + g2
        if g1 or g2:
            changes["refusal"] = resp.refusal.model_copy(update={"message": msg, "advice": adv})

    if resp.triage is not None:
        tri: dict[str, object] = {}
        msg, got = _scrub_text(resp.triage.message)
        removed += got
        if got:
            tri["message"] = msg

        items = scrub_items(resp.triage.escalation_conditions)
        if items.changed:
            removed += items.removed
            kept = list(items.items)
            # 조건을 전부 뺐다면 MONITOR 가 조건 없이 나가게 된다 (D-39 위반).
            # 안내 문장으로 자리를 메운다 — 빈 목록으로 두면 계약이 거부한다.
            tri["escalation_conditions"] = kept or [GUIDANCE]
        if tri:
            changes["triage"] = resp.triage.model_copy(update=tri)

    if not removed:
        return resp

    log.warning(
        "응답에서 연락처 %d건을 제거했다 (D-47) — %s",
        len(removed),
        " / ".join(s[:60] for s in removed),
    )
    # **더한다, 덮지 않는다.** 그래프의 `finalize` 노드가 이미 뺀 것이 있을 수 있고,
    # 그것은 여기 도착하기 전에 사라져 `removed` 에 안 잡힌다. 둘의 합이 이 응답에서
    # 실제로 빠진 문장 수다. 🔴 문장 자체는 로그에만 남긴다 — 응답에 실으면 되살아난다.
    changes["removed_contact_count"] = resp.removed_contact_count + len(removed)
    # `model_copy` 는 검증기를 돌리지 않는다. 생성자로 다시 만들어
    # 최종 안전망(`AskResponse._no_foreign_contacts`)을 반드시 태운다.
    # `full_text` 는 계산 필드라 입력에서 뺀다.
    fields = {k: getattr(resp, k) for k in AskResponse.model_fields}
    fields.update(changes)
    return AskResponse(**fields)  # type: ignore[arg-type]


class SafetyEngine:
    """`QAEngine` 을 감싸 출력 직전에 안전 조치를 건다.

    지금 거는 것은 D-47(연락처) 하나다. **여기에 더 붙일 자리**이기도 하다 —
    D-36 가명화, D-43 사진 관문 등이 같은 성질이다(모든 엔진에 공통, 출력 직전).
    """

    def __init__(self, inner: QAEngine) -> None:
        self._inner = inner

    @property
    def name(self) -> str:
        """감싼 엔진의 이름을 그대로 쓴다.

        래퍼 이름을 내보내면 `/api/health` 와 평가 리포트가 **무슨 엔진을 쟀는지**
        말하지 못한다. 래퍼는 계층이지 엔진이 아니다.
        """
        return self._inner.name

    @property
    def inner(self) -> QAEngine:
        """감싼 엔진. 테스트가 원본을 확인할 때 쓴다."""
        return self._inner

    def ask(self, req: AskRequest, session: Session) -> AskResponse:
        return scrub_response(self._inner.ask(req, session))
