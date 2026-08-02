"""평가 지표 — 순수 함수. I/O 도 엔진 의존성도 없다.

설계 근거: docs/04_테스트-평가계획.md §1.2 · §4.1 · docs/06 D-12 · D-13

    04 §1.2가 정한 것: **검색 실패와 생성 실패를 분리한다.**
    이 모듈은 그 원칙을 지표 층에서도 지킨다 —
    등급을 틀린 것과 애초에 답을 못 낸 것을 **같은 분모에 넣지 않는다.**

이 파일이 왜 엔진과 분리돼 있나
    지표 계산은 엔진 없이 단위 테스트할 수 있어야 한다.
    채점기가 틀리면 그 초록불이 곧 거짓 근거가 된다 (04 §2.5.6에서 실제로 겪었다).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

Status = Literal["answered", "clarify", "refused"]

#: 04 §4.1.0 — 숫자가 클수록 위험. `max()` 가 성립하는 방향이다 (D-39).
LEVEL_NAMES: dict[int, str] = {4: "EMERGENCY", 3: "CALL_NOW", 2: "VISIT_SOON", 1: "MONITOR"}
NAME_TO_LEVEL: dict[str, int] = {v: k for k, v in LEVEL_NAMES.items()}

#: 이 등급 이상이면 "시간이 중요한" 사례다. `missed_urgent` 의 기준.
URGENT_FLOOR = NAME_TO_LEVEL["CALL_NOW"]


def parse_level(value: str | int | None) -> int | None:
    """골든셋의 `expected_triage` 문자열을 정수 등급으로. 비었으면 None."""
    if value is None:
        return None
    if isinstance(value, int):
        return value if value in LEVEL_NAMES else None
    s = value.strip().upper()
    if not s:
        return None
    if s in NAME_TO_LEVEL:
        return NAME_TO_LEVEL[s]
    if s.isdigit() and int(s) in LEVEL_NAMES:
        return int(s)
    raise ValueError(f"알 수 없는 트리아지 등급: {value!r}")


def split_pipe(value: str | None) -> list[str]:
    """`must_cite` · `must_contain` 의 `|` 구분 필드를 목록으로."""
    if not value:
        return []
    return [p.strip() for p in value.split("|") if p.strip()]


# ─────────────────────────────────────────────────────────────
# 1건 채점
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CaseResult:
    """골든셋 1건의 채점 결과. 집계 전 단계 — 실패 분석(04 §7)이 이걸 읽는다."""

    case_id: str
    species: str
    case_type: str
    difficulty: str

    expected_status: Status
    actual_status: Status
    status_ok: bool

    #: **왜 거절했는가.** 골든셋에 `expected_refusal_reason` 열이 처음부터 있었는데
    #: 하네스가 **한 번도 읽지 않았다** (2026-08-02 1차 평가에서 발견).
    #:
    #: 그래서 *"거절했다"* 와 *"올바른 이유로 거절했다"* 가 구분되지 않았고,
    #: `범위밖` 을 기대한 건이 `되묻기상한` 으로 거절돼도 **통과**였다.
    #: `none` 유형 9건이 그렇게 100% 초록이었다 — 위장이다 (원칙 2).
    expected_refusal_reason: str = ""
    actual_refusal_reason: str = ""

    expected_level: int | None = None
    actual_level: int | None = None

    #: 근거 — must_cite 중 하나라도 실렸는가 / 전부 실렸는가
    cite_any: bool | None = None
    cite_all: bool | None = None
    cited: tuple[str, ...] = ()

    #: `must_contain` 을 **하나라도** 포함하는가. 통과 판정은 이것을 쓴다.
    contain_ok: bool | None = None
    #: `must_contain` 을 **전부** 포함하는가. 보고용 보조 지표다.
    contain_all: bool | None = None
    not_contain_ok: bool | None = None

    #: 응답 1건의 벽시계 지연(ms). 엔진이 없으면 None.
    latency_ms: float | None = None
    #: 노드별 분해 — 엔진이 제공할 때만. `{"retrieve": 193.0, "generate": 8200.0}`
    node_ms: tuple[tuple[str, float], ...] = ()

    #: 예외로 죽은 경우. 계약 위반(pydantic ValidationError)도 여기 잡힌다.
    error: str | None = None

    @property
    def level_delta(self) -> int | None:
        """예측 − 정답. 음수면 과소평가다. 한쪽이라도 없으면 None."""
        if self.expected_level is None or self.actual_level is None:
            return None
        return self.actual_level - self.expected_level

    @property
    def under(self) -> bool:
        d = self.level_delta
        return d is not None and d < 0

    @property
    def over(self) -> bool:
        d = self.level_delta
        return d is not None and d > 0

    @property
    def critical_under(self) -> bool:
        """중대 과소평가 — 정답 EMERGENCY(4) 인데 예측이 2 이하 (04 §4.1.0). 목표 0."""
        return self.expected_level == 4 and self.actual_level is not None and self.actual_level <= 2

    @property
    def missed_urgent(self) -> bool:
        """정답이 CALL_NOW 이상인데 **등급을 아예 못 냈다** (거절·되묻기).

        등급 오류가 아니므로 과소평가율 분모에 넣지 않는다 (§1.2).
        그러나 사용자가 받는 결과는 "긴급도를 못 들었다" 이므로 **따로 반드시 센다.**
        숨기면 과소평가율만 예쁘게 나온다.
        """
        return (
            self.expected_level is not None
            and self.expected_level >= URGENT_FLOOR
            and self.actual_level is None
        )

    @property
    def reason_ok(self) -> bool | None:
        """거절 이유가 맞는가. 기대값이 없으면 `None`(채점 안 함).

        **`refused` 일 때만 본다.** 되묻기·정상 응답에는 거절 이유가 없다 —
        없는 것을 틀렸다고 세면 분모가 오염된다 (04 §1.2).
        """
        if not self.expected_refusal_reason or self.actual_status != "refused":
            return None
        return self.actual_refusal_reason == self.expected_refusal_reason

    @property
    def passed(self) -> bool:
        """이 건이 통과인가 — 상태·**거절 이유**·등급·근거·문구를 모두 만족해야 한다."""
        if self.error is not None or not self.status_ok:
            return False
        if self.reason_ok is False:
            return False
        if self.expected_level is not None and self.actual_level != self.expected_level:
            return False
        for flag in (self.cite_any, self.contain_ok, self.not_contain_ok):
            if flag is False:
                return False
        return True


def score_case(
    row: dict[str, str],
    *,
    status: Status | None,
    level: int | None,
    refusal_reason: str = "",
    answer_text: str,
    citations: Sequence[str],
    latency_ms: float | None = None,
    node_ms: dict[str, float] | None = None,
    error: str | None = None,
) -> CaseResult:
    """골든셋 행 + 엔진 응답 → 채점 결과 1건.

    `answer_text` 는 **`full_text` 를 넣는다.** `answer` 만 보면 상승 조건이
    빠진 문장을 채점하게 되고, 조건 누락은 이 도메인에서 과소평가와 같다 (D-39).
    """
    expected_status = (row.get("expected_status") or "").strip()
    expected_level = parse_level(row.get("expected_triage"))
    expected_reason = (row.get("expected_refusal_reason") or "").strip()

    must_cite = split_pipe(row.get("must_cite"))
    must_contain = split_pipe(row.get("must_contain"))
    must_not = split_pipe(row.get("must_not_contain"))

    cite_any = cite_all = None
    if must_cite:
        got = set(citations)
        cite_any = any(s in got for s in must_cite)
        cite_all = all(s in got for s in must_cite)

    # `must_contain` 은 **any** 로 본다.
    #
    # 골든셋 52건 중 36건이 `|` 를 썼고, 내용을 보면 전부 **선택지**다 —
    # `어떤 초콜릿|종류|다크|밀크`(G-047) 는 넷 중 하나만 나와도 되묻기가 성립한다.
    # `all` 로 채점하던 때는 이런 케이스가 **구조적으로 통과 불가**였다 (2026-08-02).
    # `must_cite` 가 any/all 두 지표를 다 내는 것과 같은 방식으로, 여기도 둘 다 낸다.
    contain_any = any(k in answer_text for k in must_contain) if must_contain else None
    contain_all = all(k in answer_text for k in must_contain) if must_contain else None
    # 금지 문구는 반대다 — **하나라도 있으면 실패**다. `all/any` 선택의 여지가 없다.
    not_contain_ok = not any(k in answer_text for k in must_not) if must_not else None

    return CaseResult(
        case_id=row.get("case_id", ""),
        species=row.get("species") or "(미지정)",
        case_type=row.get("case_type") or "(미분류)",
        difficulty=row.get("difficulty") or "",
        expected_status=expected_status,  # type: ignore[arg-type]
        actual_status=status,  # type: ignore[arg-type]
        status_ok=(status == expected_status),
        expected_refusal_reason=expected_reason,
        actual_refusal_reason=(refusal_reason or "").strip(),
        expected_level=expected_level,
        actual_level=level,
        cite_any=cite_any,
        cite_all=cite_all,
        cited=tuple(citations),
        contain_ok=contain_any,
        contain_all=contain_all,
        not_contain_ok=not_contain_ok,
        latency_ms=latency_ms,
        node_ms=tuple(sorted((node_ms or {}).items())),
        error=error,
    )


# ─────────────────────────────────────────────────────────────
# 집계
# ─────────────────────────────────────────────────────────────
def _rate(num: int, den: int) -> float | None:
    """분모가 0이면 0.0 이 아니라 **None** 이다.

    0.0 으로 채우면 "측정했는데 완벽했다" 로 읽힌다.
    측정하지 않은 것과 측정해서 0인 것은 다르다 — 04 §2.5.6이 겪은 실수다.
    """
    return None if den == 0 else num / den


def percentile(values: Sequence[float], p: float) -> float | None:
    """오름차순 nearest-rank. 표본이 비면 **None** (`_rate` 와 같은 규칙).

    보간하지 않는 이유 — 골든셋이 52건이라 표본이 작다. 보간하면 **실제로 관측되지
    않은 값**이 지표로 나가고, 이 프로젝트가 계속 경계해 온 "만든 숫자"가 된다.
    실제 응답 하나의 지연을 그대로 돌려준다.
    """
    if not values:
        return None
    ordered = sorted(values)
    k = max(1, math.ceil(p * len(ordered)))
    return ordered[k - 1]


@dataclass
class Summary:
    """집계 지표. 분모를 항상 함께 들고 다닌다 — 비율만 보면 표본 1건도 100%다."""

    n: int = 0
    errors: int = 0
    passed: int = 0

    status_correct: int = 0

    #: 등급이 **양쪽 다 있는** 건만. 04 §1.2의 분리 원칙.
    level_n: int = 0
    level_exact: int = 0
    level_adjacent: int = 0  # |예측 − 정답| ≤ 1
    under: int = 0
    over: int = 0
    critical_under: int = 0

    #: 등급 오류가 아니라 **판정 자체를 못 낸** 긴급 건. 분모가 다르므로 따로 센다.
    urgent_n: int = 0
    missed_urgent: int = 0

    cite_n: int = 0
    cite_any: int = 0
    cite_all: int = 0

    contain_n: int = 0
    contain_ok: int = 0
    contain_all: int = 0

    #: 금지 문구는 **답을 낸 건만** 분모에 넣는다.
    #:
    #: 예전에는 상태와 무관하게 셌다. 스텁 실행에서 분모 38건의 구성이
    #: 거절 30 · 되묻기 7 · answered 1 이었고, 결과가 **100.0% (38/38)** 로 찍혔다.
    #: *"답을 안 해서 금지어를 안 썼다"* 가 만점으로 보고된 것이다 (2026-08-02).
    #: `missed_urgent` 는 분모를 분리했는데 문구 지표만 안 했다.
    not_contain_n: int = 0
    not_contain_ok: int = 0
    #: 전체(거절·되묻기 포함) 분모. 위 값과 나란히 보여 준다.
    not_contain_all_n: int = 0
    not_contain_all_ok: int = 0

    confusion: Counter = field(default_factory=Counter)  # (정답등급, 예측등급)
    status_confusion: Counter = field(default_factory=Counter)

    #: 지연. **`answered` 를 따로 모은다** — 되묻기·거절은 LLM 을 2번만 돌아 빠르고,
    #: 섞으면 평균이 낙관적으로 나온다. 긴 답변을 받는 사람의 경험이 가려진다.
    latencies: list[float] = field(default_factory=list)
    answered_latencies: list[float] = field(default_factory=list)
    node_totals: dict[str, list[float]] = field(default_factory=dict)

    # ── 비율 ────────────────────────────────────────────
    @property
    def pass_rate(self) -> float | None:
        return _rate(self.passed, self.n)

    @property
    def status_accuracy(self) -> float | None:
        return _rate(self.status_correct, self.n)

    @property
    def level_accuracy(self) -> float | None:
        return _rate(self.level_exact, self.level_n)

    @property
    def adjacent_accuracy(self) -> float | None:
        return _rate(self.level_adjacent, self.level_n)

    @property
    def under_rate(self) -> float | None:
        """🔴 과소평가율 — 04 §4.1의 최우선 지표."""
        return _rate(self.under, self.level_n)

    @property
    def over_rate(self) -> float | None:
        """과대평가율. **낮다고 좋은 게 아니다** — D-50 매핑이 의도한 편향이 있다 (04 §4.1.x)."""
        return _rate(self.over, self.level_n)

    @property
    def critical_under_rate(self) -> float | None:
        return _rate(self.critical_under, self.level_n)

    @property
    def missed_urgent_rate(self) -> float | None:
        return _rate(self.missed_urgent, self.urgent_n)

    @property
    def cite_any_rate(self) -> float | None:
        return _rate(self.cite_any, self.cite_n)

    @property
    def cite_all_rate(self) -> float | None:
        return _rate(self.cite_all, self.cite_n)

    @property
    def contain_rate(self) -> float | None:
        """`must_contain` 중 **하나라도** 포함한 비율."""
        return _rate(self.contain_ok, self.contain_n)

    @property
    def contain_all_rate(self) -> float | None:
        """`must_contain` 을 **전부** 포함한 비율. 참고용."""
        return _rate(self.contain_all, self.contain_n)

    @property
    def not_contain_rate(self) -> float | None:
        """**답을 낸 건**에서 금지 문구를 피한 비율. 이것이 진짜 지표다."""
        return _rate(self.not_contain_ok, self.not_contain_n)

    @property
    def not_contain_rate_all(self) -> float | None:
        """전체(거절·되묻기 포함) 기준. 낙관적으로 보이므로 함께만 읽는다."""
        return _rate(self.not_contain_all_ok, self.not_contain_all_n)

    # ── 지연 ────────────────────────────────────────────
    @property
    def p50_ms(self) -> float | None:
        return percentile(self.latencies, 0.50)

    @property
    def p95_ms(self) -> float | None:
        """**p95 를 본다.** 평균은 되묻기·거절이 끌어내려 낙관적이다."""
        return percentile(self.latencies, 0.95)

    @property
    def answered_p50_ms(self) -> float | None:
        return percentile(self.answered_latencies, 0.50)

    @property
    def answered_p95_ms(self) -> float | None:
        """긴 답변을 받는 사람의 경험. **여기가 실제 체감이다** —
        02 §12.4 로 스트리밍을 미채택했으므로 이 지연이 그대로 침묵으로 나타난다."""
        return percentile(self.answered_latencies, 0.95)

    def node_p95(self) -> list[tuple[str, float]]:
        """노드별 p95, 느린 순. **어디가 느린지 모르면 캐시를 붙여도 소용없다.**"""
        out = [(n, percentile(v, 0.95)) for n, v in self.node_totals.items()]
        return sorted(((n, ms) for n, ms in out if ms is not None), key=lambda kv: -kv[1])


def summarize(results: Iterable[CaseResult]) -> Summary:
    s = Summary()
    for r in results:
        s.n += 1
        if r.error is not None:
            s.errors += 1
        if r.passed:
            s.passed += 1
        if r.status_ok:
            s.status_correct += 1
        s.status_confusion[(r.expected_status, r.actual_status)] += 1

        if r.expected_level is not None and r.expected_level >= URGENT_FLOOR:
            s.urgent_n += 1
            if r.missed_urgent:
                s.missed_urgent += 1

        d = r.level_delta
        if d is not None:
            s.level_n += 1
            s.confusion[(r.expected_level, r.actual_level)] += 1
            if d == 0:
                s.level_exact += 1
            if abs(d) <= 1:
                s.level_adjacent += 1
            if d < 0:
                s.under += 1
            if d > 0:
                s.over += 1
            if r.critical_under:
                s.critical_under += 1

        if r.cite_any is not None:
            s.cite_n += 1
            s.cite_any += int(r.cite_any)
            s.cite_all += int(bool(r.cite_all))
        if r.contain_ok is not None:
            s.contain_n += 1
            s.contain_ok += int(r.contain_ok)
            s.contain_all += int(bool(r.contain_all))
        if r.not_contain_ok is not None:
            s.not_contain_all_n += 1
            s.not_contain_all_ok += int(r.not_contain_ok)
            # 답을 낸 건만 진짜 분모다 (필드 주석 참조)
            if r.actual_status == "answered":
                s.not_contain_n += 1
                s.not_contain_ok += int(r.not_contain_ok)

        if r.latency_ms is not None:
            s.latencies.append(r.latency_ms)
            if r.actual_status == "answered":
                s.answered_latencies.append(r.latency_ms)
        for node, ms in r.node_ms:
            s.node_totals.setdefault(node, []).append(ms)
    return s


def group_by(results: Sequence[CaseResult], key: str) -> dict[str, Summary]:
    """종별·유형별 분리 집계 (04 §4.2).

    **전체 평균만 보면 조류 성능 저하가 가려진다** — 그래서 이 함수가 있다.
    """
    buckets: dict[str, list[CaseResult]] = {}
    for r in results:
        buckets.setdefault(getattr(r, key), []).append(r)
    return {k: summarize(v) for k, v in sorted(buckets.items())}


def fmt_ms(value: float | None) -> str:
    """지연 표기. `None` 은 `—` — 안 쟀다는 뜻이지 0 이 아니다."""
    if value is None:
        return "—"
    return f"{value / 1000:.2f}s" if value >= 1000 else f"{value:.0f}ms"


def fmt(value: float | None, *, pct: bool = True) -> str:
    """None 은 `—` 로 찍는다. 0.0 과 구분되어야 한다."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"
