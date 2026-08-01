"""평가 지표 회귀 테스트.

**채점기가 틀리면 그 초록불이 곧 거짓 근거다** — 04 §2.5.6에서 실제로 겪었다.
`build_index.py` 의 정답 목록이 별칭을 몰라 "상위에 없음(정상)" 을 찍었는데,
사실은 부동액이 4위에 있었다. 그래서 지표 코드를 엔진과 분리해 따로 고정한다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval" / "harness"))

from metrics import (  # noqa: E402
    URGENT_FLOOR,
    Summary,
    fmt,
    group_by,
    parse_level,
    score_case,
    split_pipe,
    summarize,
)


def row(**kw) -> dict[str, str]:
    base = {
        "case_id": "G-000",
        "species": "dog",
        "case_type": "dose",
        "difficulty": "보통",
        "expected_status": "answered",
        "expected_triage": "",
        "must_cite": "",
        "must_contain": "",
        "must_not_contain": "",
    }
    base.update(kw)
    return base


# ── 파싱 ────────────────────────────────────────────────
@pytest.mark.parametrize(
    "raw,want",
    [
        ("EMERGENCY", 4),
        ("CALL_NOW", 3),
        ("VISIT_SOON", 2),
        ("MONITOR", 1),
        ("", None),
        (None, None),
    ],
)
def test_parse_level(raw, want):
    assert parse_level(raw) == want


def test_parse_level_rejects_unknown():
    """오타를 조용히 None 으로 삼키면 그 케이스가 채점에서 사라진다."""
    with pytest.raises(ValueError):
        parse_level("URGENT")


def test_split_pipe():
    assert split_pipe("S-034|S-021") == ["S-034", "S-021"]
    assert split_pipe("") == []


# ── 방향이 있는 오류 ────────────────────────────────────
def test_under_triage_is_directional():
    """정답 EMERGENCY(4) → 예측 MONITOR(1) 은 과소평가이고 중대 과소평가다 (D-13)."""
    r = score_case(
        row(expected_triage="EMERGENCY"), status="answered", level=1, answer_text="", citations=[]
    )
    assert r.under and not r.over
    assert r.critical_under
    assert r.level_delta == -3


def test_over_triage_is_not_critical():
    r = score_case(
        row(expected_triage="MONITOR"), status="answered", level=4, answer_text="", citations=[]
    )
    assert r.over and not r.under and not r.critical_under


def test_critical_under_boundary():
    """EMERGENCY(4) → CALL_NOW(3) 은 과소평가지만 **중대는 아니다** (기준: 예측 ≤ 2)."""
    r = score_case(
        row(expected_triage="EMERGENCY"), status="answered", level=3, answer_text="", citations=[]
    )
    assert r.under and not r.critical_under


# ── 04 §1.2 — 분모를 섞지 않는다 ────────────────────────
def test_refusal_on_urgent_case_is_not_under_triage():
    """정답이 EMERGENCY 인데 거절했다면 **등급 오류가 아니다.**

    과소평가율 분모에 넣으면 등급 혼동행렬이 오염된다.
    대신 `missed_urgent` 로 반드시 따로 센다 — 숨기면 과소평가율만 예뻐진다.
    """
    r = score_case(
        row(expected_status="refused", expected_triage="EMERGENCY"),
        status="refused",
        level=None,
        answer_text="",
        citations=[],
    )
    assert r.level_delta is None
    assert not r.under
    assert r.missed_urgent

    s = summarize([r])
    assert s.level_n == 0
    assert s.under_rate is None  # 0.0 이 아니다
    assert s.missed_urgent == 1 and s.urgent_n == 1


def test_missed_urgent_floor():
    """VISIT_SOON(2) 은 긴급 기준 미만이라 `missed_urgent` 로 세지 않는다."""
    assert URGENT_FLOOR == 3
    r = score_case(
        row(expected_triage="VISIT_SOON"),
        status="refused",
        level=None,
        answer_text="",
        citations=[],
    )
    assert not r.missed_urgent


def test_empty_denominator_is_none_not_zero():
    """측정하지 않은 것과 측정해서 0인 것은 다르다."""
    assert Summary().under_rate is None
    assert fmt(None) == "—"
    assert fmt(0.0) == "0.0%"


# ── 근거·문구 ───────────────────────────────────────────
def test_must_cite_any_and_all():
    r = score_case(
        row(must_cite="S-034|S-021"),
        status="answered",
        level=None,
        answer_text="",
        citations=["S-034"],
    )
    assert r.cite_any and not r.cite_all


def test_must_not_contain_blocks_pass():
    """`must_not_contain` 이 걸리면 등급이 맞아도 통과가 아니다.

    G-045(개 백합)가 이 경로를 쓴다 — 등급은 CALL_NOW 로 맞아도
    답변에 '신부전' 이 있으면 고양이 기준을 옮긴 것이다 (D-09).
    """
    r = score_case(
        row(expected_triage="CALL_NOW", must_cite="S-096", must_not_contain="신부전|신독성"),
        status="answered",
        level=3,
        answer_text="개가 백합을 먹으면 급성 신부전이 올 수 있습니다.",
        citations=["S-096"],
    )
    assert r.status_ok and r.actual_level == r.expected_level
    assert r.not_contain_ok is False
    assert not r.passed


def test_pass_requires_everything():
    r = score_case(
        row(expected_triage="CALL_NOW", must_cite="S-034", must_contain="수의사"),
        status="answered",
        level=3,
        answer_text="지금 수의사에게 전화해 상태를 알리세요.",
        citations=["S-034"],
    )
    assert r.passed


def test_error_case_never_passes():
    r = score_case(
        row(), status=None, level=None, answer_text="", citations=[], error="ValidationError: ..."
    )
    assert not r.passed
    assert summarize([r]).errors == 1


# ── 집계 ────────────────────────────────────────────────
def test_group_by_species_separates():
    """전체 평균만 보면 조류 저하가 가려진다 (04 §4.2)."""
    rs = [
        score_case(
            row(species="dog", expected_triage="CALL_NOW"),
            status="answered",
            level=3,
            answer_text="",
            citations=[],
        ),
        score_case(
            row(species="bird", expected_triage="EMERGENCY"),
            status="answered",
            level=1,
            answer_text="",
            citations=[],
        ),
    ]
    g = group_by(rs, "species")
    assert g["dog"].under_rate == 0.0
    assert g["bird"].under_rate == 1.0
    assert summarize(rs).under_rate == 0.5  # 평균은 조류 실패를 절반으로 희석한다


def test_adjacent_accuracy():
    rs = [
        score_case(
            row(expected_triage="CALL_NOW"),
            status="answered",
            level=lv,
            answer_text="",
            citations=[],
        )
        for lv in (3, 4, 1)  # 정확 / 인접 / 2단계 하락
    ]
    s = summarize(rs)
    assert s.level_exact == 1 and s.level_adjacent == 2
    assert s.under == 1 and s.over == 1


# ── 고지 문구를 채점하지 않는다 ─────────────────────────
def test_disclaimer_is_excluded_from_scoring():
    """고정 상용구를 채점하면 지표가 상용구를 측정한다.

    `DISCLAIMER` 는 모든 응답에 무조건 붙는다 (02 §9). 그 문장 안에
    '진단'·'수의사'가 들어 있어서, `full_text` 를 그대로 채점하면

      · must_contain: 수의사    → 거절 응답도 거저 통과
      · must_not_contain: 진단  → 어떤 응답도 통과 불가 (G-004가 실제로 걸렸다)

    `run_eval.scored_text()` 가 고지를 빼고 상승 조건은 남기는지 고정한다.
    """
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from run_eval import scored_text

    from pettriage.app.contracts import DISCLAIMER

    refused = SimpleNamespace(
        answer=None,
        clarify=None,
        refusal=SimpleNamespace(message="근거를 찾을 수 없습니다.", advice="수의사와 상담하세요."),
        triage=None,
    )
    text = scored_text(refused)
    assert DISCLAIMER not in text
    assert "진단" not in text  # 고지의 '진단'이 새어 들어오면 G-004가 영원히 실패한다

    answered = SimpleNamespace(
        answer="지금 수의사에게 전화해 상태를 알리세요.",
        clarify=None,
        refusal=None,
        triage=SimpleNamespace(escalation_conditions=["구토", "발작"]),
    )
    text = scored_text(answered)
    assert "발작" in text  # 상승 조건은 남아야 한다 (D-39)
    assert DISCLAIMER not in text


# ── 지연 ────────────────────────────────────────────────
def test_percentile_is_nearest_rank_not_interpolated():
    """보간하지 않는다. 골든셋이 52건이라 표본이 작아,
    보간하면 **실제로 관측되지 않은 지연**이 지표로 나간다."""
    from metrics import percentile

    vals = [10, 20, 30, 40, 50]
    assert percentile(vals, 0.50) == 30
    assert percentile(vals, 0.95) == 50
    assert percentile(vals, 0.0) == 10  # k 는 최소 1


def test_percentile_empty_is_none():
    from metrics import percentile

    assert percentile([], 0.95) is None
    assert Summary().p95_ms is None
    assert Summary().answered_p95_ms is None


def test_answered_latency_is_separated():
    """되묻기·거절은 LLM 을 2번만 돌아 빠르다. 섞으면 평균이 낙관적이다."""
    rs = [
        score_case(
            row(expected_status="clarify"),
            status="clarify",
            level=None,
            answer_text="",
            citations=[],
            latency_ms=200.0,
        ),
        score_case(
            row(expected_status="refused"),
            status="refused",
            level=None,
            answer_text="",
            citations=[],
            latency_ms=300.0,
        ),
        score_case(
            row(expected_status="answered", must_cite="S-034"),
            status="answered",
            level=None,
            answer_text="",
            citations=["S-034"],
            latency_ms=20_000.0,
        ),
    ]
    s = summarize(rs)
    assert s.p50_ms == 300.0  # 전체는 빠른 쪽이 끌어내린다
    assert s.answered_p50_ms == 20_000.0  # answered 만 보면 실제 체감이 드러난다
    assert s.answered_p95_ms == 20_000.0


def test_node_p95_sorted_slowest_first():
    """어디가 느린지 모르면 캐시를 붙여도 소용없다."""
    rs = [
        score_case(
            row(),
            status="answered",
            level=None,
            answer_text="",
            citations=[],
            latency_ms=1000.0,
            node_ms={"retrieve": 193.0, "generate": 8000.0},
        ),
        score_case(
            row(),
            status="answered",
            level=None,
            answer_text="",
            citations=[],
            latency_ms=1200.0,
            node_ms={"retrieve": 210.0, "generate": 9000.0},
        ),
    ]
    ranked = summarize(rs).node_p95()
    assert [n for n, _ in ranked] == ["generate", "retrieve"]
    assert ranked[0][1] == 9000.0


def test_fmt_ms_distinguishes_unmeasured_from_zero():
    from metrics import fmt_ms

    assert fmt_ms(None) == "—"  # 안 쟀다
    assert fmt_ms(0) == "0ms"  # 쟀는데 0이다
    assert fmt_ms(193) == "193ms"
    assert fmt_ms(8200) == "8.20s"
