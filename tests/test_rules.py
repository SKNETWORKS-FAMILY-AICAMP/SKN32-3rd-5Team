"""규칙 테이블 조회 (D-16 · D-39 · D-46).

이 테스트가 지키는 것은 **표의 값**이 아니라 **표를 읽는 방식**이다.
값은 사실 표에서 파생되므로 바뀔 수 있지만, 읽는 규칙이 바뀌면 판정이 틀린다.
"""

from __future__ import annotations

import pytest

from pettriage.compute.rules import (
    COMPUTABLE_UNITS,
    SEVERITY,
    Rule,
    computable_for,
    has_quantitative,
    load_rules,
    lookup,
    parse_low,
    qualitative_for,
    rule_level_for,
    to_mg_per_kg,
)
from pettriage.triage.levels import TriageLevel


class TestParseLow:
    """**범위와 부등호는 낮은 쪽으로 읽는다.**

    높은 쪽을 쓰면 40 mg/kg 을 먹은 개가 "아직 안전"으로 나온다.
    """

    @pytest.mark.parametrize(
        ("dose", "expect"),
        [
            ("20", 20.0),
            ("40-50", 40.0),
            ("≥1", 1.0),
            ("2-2.5", 2.0),
            ("0.03", 0.03),
            ("15-30", 15.0),
            ("", None),
            ("불명", None),
        ],
    )
    def test_낮은_값을_고른다(self, dose: str, expect: float | None) -> None:
        assert parse_low(dose) == expect


class TestLoad:
    def test_표가_비어_있지_않다(self) -> None:
        assert len(load_rules()) >= 10

    def test_조류는_한_행도_없다(self) -> None:
        """D-09 — 코퍼스에 쓸 수 있는 조류 체중당 역치가 0건이다.

        여기에 조류 행이 생기면 **D-09 종별 분기를 다시 검토해야 한다.**
        """
        assert not [r for r in load_rules() if r.species == "bird"]

    def test_심각도_어휘가_스키마와_맞는다(self) -> None:
        from pettriage.ingest.templates import THRESHOLD_TYPES

        assert set(SEVERITY) == set(THRESHOLD_TYPES)
        assert {r.threshold_type for r in load_rules()} <= set(THRESHOLD_TYPES)


class TestLookup:
    def test_부분_일치로_찾는다(self) -> None:
        """표는 `초콜릿(테오브로민+카페인)`, 질의는 `초콜릿` 이다."""
        assert computable_for("초콜릿", "dog")

    def test_종을_넓혀서_본다(self) -> None:
        """마늘은 `mammal` 행이다 — `dog` 질의에 걸려야 한다 (D-39)."""
        assert [r for r in lookup("마늘", "dog") if r.species == "mammal"]

    def test_종이_다르면_안_걸린다(self) -> None:
        """란타나는 `dog` 행뿐이다. 고양이에게 개 수치를 주면 안 된다."""
        assert not [r for r in lookup("란타나", "cat") if r.species == "dog"]

    def test_중복_출처를_접는다(self) -> None:
        """양파 `15-30 g/kg` 가 S-034·S-098 에 같은 값으로 있다."""
        rules = lookup("양파", "dog")
        doses = [(r.substance, r.dose, r.unit) for r in rules]
        assert len(doses) == len(set(doses)), f"중복이 남았다: {doses}"

    def test_없으면_빈_리스트다(self) -> None:
        """**지어내지 않는다.**"""
        assert lookup("존재하지않는물질", "dog") == []
        assert computable_for("초콜릿", "bird") == []

    def test_낮은_역치부터_나온다(self) -> None:
        """초콜릿 개 — `임상징후 발현 20` → `중증 40-50` → `중증 60`."""
        got = [(r.threshold_type, r.low) for r in computable_for("초콜릿", "dog")]
        assert got == sorted(got, key=lambda t: (SEVERITY[t[0]], t[1]))
        assert got[0][1] == 20.0


class TestComputableGate:
    """**계산 불가 행이 계산에 새어 들어가면 안 된다.**"""

    def test_백합은_계산에서_빠진다(self) -> None:
        """원문이 *"one or two leaves"* 로만 말했다 — 잎 무게를 우리가 정할 수 없다."""
        assert not computable_for("백합", "cat")
        assert [r.unit for r in qualitative_for("백합", "cat")] == ["leaves"]

    def test_주목은_계산에서_빠진다(self) -> None:
        """단위가 `g leaves/kg` 다. `g/kg` 로 읽으면 식물 전체 무게로 오독한다."""
        rules = qualitative_for("주목", "dog")
        assert rules and rules[0].unit == "g leaves/kg"
        assert not computable_for("주목", "dog")

    def test_계산_가능_행은_전부_환산된다(self) -> None:
        """`computable=Y` 인 행은 **반드시 `mg/kg` 으로 환산된다.**

        ⚠️ 이 테스트는 2026-08-02 에 목록을 직접 들지 않도록 바뀌었다.

            ok = {"mg/kg", "g/kg", "mL/kg", "%"}      # ← 자기 목록을 갖고 있었다

        `mL/kg` 이 여기 있었기 때문에, 빌더와 환산표가 어긋난 상태
        (빌더는 계산 가능, `_MG_PER_KG` 에는 없음)를 **테스트가 통과시켰다.**
        목록을 세 곳(빌더·리더·테스트)에 두면 테스트가 어긋남을 못 잡는다.
        이제 단일 출처를 보고, 환산이 실제로 되는지까지 확인한다.
        """
        for r in load_rules():
            if not r.computable:
                continue
            assert (
                r.unit in COMPUTABLE_UNITS
            ), f"{r.fact_id} 단위 {r.unit!r} 가 계산 가능으로 표시됐다"
            assert (
                to_mg_per_kg(r.low, r.unit) is not None
            ), f"{r.fact_id} 단위 {r.unit!r} 가 환산되지 않는다"

    def test_역치가_전부_환산불가면_크래시하지_않는다(self) -> None:
        """`min()` 을 빈 시퀀스로 부르면 터진다. 정성 답변으로 내려가야 한다.

        빌더가 `computable=Y` 로 표시했는데 리더가 환산을 못 하는 상황 —
        두 목록이 어긋나면 언제든 다시 생긴다. 그때 크래시가 아니라
        `level=None`(= LLM 판정에 맡긴다)이 나오는지 확인한다.
        """
        import pettriage.compute.rules as rules_mod

        fake = Rule(
            fact_id="F-TEST-001",
            substance="가상물질",
            species="dog",
            threshold_type="임상징후 발현",
            dose="5",
            unit="mL/kg",  # 환산표에 없다 — 밀도를 모르면 질량으로 못 바꾼다
            computable=True,
            effect="",
            signs="구토",
            onset="",
            source_id="S-TEST",
            citation="",
            note="",
        )
        orig = rules_mod.computable_for
        rules_mod.computable_for = lambda s, sp: [fake]  # type: ignore[assignment]
        try:
            v = rules_mod.rule_level_for("가상물질", "dog", 1000.0)
        finally:
            rules_mod.computable_for = orig  # type: ignore[assignment]
        assert v.level is None
        assert "환산표에 없다" in v.reason

    def test_계산_가능_행은_수치가_있다(self) -> None:
        for r in load_rules():
            if r.computable:
                assert r.low is not None, f"{r.fact_id} dose={r.dose!r}"


class TestHasQuantitative:
    """D-46 — 정량 질의에 정량 근거가 없다는 판정은 **검색이 아니라 여기 일이다.**"""

    def test_개_초콜릿은_정량_가능(self) -> None:
        assert has_quantitative("초콜릿", "dog")

    def test_앵무새_초콜릿은_정량_불가(self) -> None:
        """검색은 조류 초콜릿 청크를 잘 물어온다. 그래도 **수치는 없다.**"""
        assert not has_quantitative("초콜릿", "bird")

    def test_고양이_백합은_정량_불가(self) -> None:
        """근거는 있으나 개수 단위다 — 정성 답변으로 내려가야 한다."""
        assert not has_quantitative("백합", "cat")
        assert qualitative_for("백합", "cat")


class TestRuleLevel:
    """바닥 등급 산출 (D-50).

    **`rule_level` 은 정밀한 판정이 아니라 바닥이다** — `final = max(rule, llm)` 이므로
    LLM 은 올릴 수만 있다 (D-09). 그래서 이 테스트가 지키는 것은
    "등급이 의학적으로 맞나"가 아니라 **"틀렸을 때 어느 쪽으로 틀리나"** 다.
    """

    def test_역치_미만은_MONITOR_에_상승조건이_붙는다(self) -> None:
        """`None` 으로 두면 *"조금 먹었는데요"* 가 전부 거절이 된다.

        가장 흔한 질의가 가장 잘 깨지는 설계가 된다.
        """
        v = rule_level_for("초콜릿", "dog", 5)
        assert v.level is TriageLevel.MONITOR
        assert v.escalation_conditions, "상승 조건이 없으면 apply_gate 가 거부한다 (D-39)"

    def test_임상징후_발현은_CALL_NOW(self) -> None:
        """계산 가능한 12행 중 9행이 여기다 — **사실상 기본 등급**이다."""
        assert rule_level_for("초콜릿", "dog", 25).level is TriageLevel.CALL_NOW

    def test_중증은_EMERGENCY(self) -> None:
        """초콜릿 40-50 mg/kg 은 4kg 개가 다크초콜릿 20g — 경련·부정맥 구간이다."""
        assert rule_level_for("초콜릿", "dog", 45).level is TriageLevel.EMERGENCY

    def test_여러_역치를_넘기면_가장_높은_등급(self) -> None:
        v = rule_level_for("초콜릿", "dog", 1000)
        assert v.level is TriageLevel.EMERGENCY
        assert len(v.crossed) >= 2

    def test_상향_여지가_남아_있다(self) -> None:
        """바닥이 천장이면 D-09 게이트가 할 일이 없어진다.

        `임상징후 발현`(9행)이 `CALL_NOW` 라 LLM 이 `EMERGENCY` 로 올릴 수 있다.
        """
        assert rule_level_for("초콜릿", "dog", 25).level < TriageLevel.EMERGENCY

    def test_조류는_바닥을_만들지_못한다(self) -> None:
        """코퍼스에 조류 체중당 역치가 0건이다 (D-09). **수치를 지어내면 그게 환각이다.**"""
        v = rule_level_for("초콜릿", "bird", 10_000)
        assert v.level is None
        assert "역치가 없다" in v.reason

    def test_근거를_함께_돌려준다(self) -> None:
        """답변이 어느 행을 근거로 삼았는지 남아야 추적이 된다 (04 §7)."""
        v = rule_level_for("초콜릿", "dog", 25)
        assert v.crossed and v.crossed[0].fact_id.startswith("F-")
        assert "F-" in v.reason


class TestUnitNormalization:
    """`%` 는 체중 대비 백분율이다 — 1% = 10 g/kg = 10,000 mg/kg."""

    def test_환산(self) -> None:
        assert to_mg_per_kg(1, "mg/kg") == pytest.approx(1)
        assert to_mg_per_kg(1, "g/kg") == pytest.approx(1000)
        assert to_mg_per_kg(1, "%") == pytest.approx(10_000)

    def test_모르는_단위는_None(self) -> None:
        assert to_mg_per_kg(1, "seeds") is None

    def test_서로_다른_출처가_같은_값으로_수렴한다(self) -> None:
        """**우연이 아니라 검증이다.**

        S-014 는 알리움류를 `0.5%`, S-034·S-098 은 양파를 `5 g/kg` 로 적는다.
        환산하면 둘 다 5,000 mg/kg 이다 — 서로 다른 자료가 같은 값을 말한다.
        어긋나면 `CONFLICT_RATIO` 가 잡는다.
        """
        assert to_mg_per_kg(0.5, "%") == pytest.approx(to_mg_per_kg(5, "g/kg"))
        assert rule_level_for("양파", "cat", 6000).level is TriageLevel.CALL_NOW
        assert rule_level_for("양파", "cat", 2000).level is TriageLevel.MONITOR


class TestConflict:
    """출처 간 수치가 10배 이상 벌어지면 **정량 판정을 포기한다** (D-50).

    S-034 는 건포도를 본문 `2.8 mg/kg`, 같은 논문 표 `2.8-36.4 g/kg` 로 적는다 —
    **1,000배 차이**다. 낮은 쪽을 바닥으로 쓰는 원칙만 두면
    단위 오류가 섞인 순간 **거의 모든 섭취가 역치 초과**가 된다.
    """

    def _v(self, doses: list[tuple[str, str]]):
        from pettriage.compute.rules import Rule, _detect_conflict

        rules = [
            Rule(
                fact_id=f"F-T-{i:03}",
                substance="테스트물질",
                species="dog",
                threshold_type="임상징후 발현",
                dose=d,
                unit=u,
                computable=True,
                effect="",
                signs="구토",
                onset="",
                source_id=f"S-{i:03}",
                citation="",
                note="",
            )
            for i, (d, u) in enumerate(doses)
        ]
        return _detect_conflict(rules)

    def test_1000배_차이는_포기한다(self) -> None:
        assert self._v([("2.8", "mg/kg"), ("2.8", "g/kg")]) is not None

    def test_반올림_차이는_포기하지_않는다(self) -> None:
        """`20` 과 `25` 는 자료가 반올림을 다르게 한 것이지 상충이 아니다."""
        assert self._v([("20", "mg/kg"), ("25", "mg/kg")]) is None

    def test_정확히_10배는_포기한다(self) -> None:
        assert self._v([("2", "mg/kg"), ("20", "mg/kg")]) is not None

    def test_역치종류가_다르면_비교하지_않는다(self) -> None:
        """`임상징후 발현 20` 과 `중증 60` 은 **상충이 아니라 단계다.**"""
        assert rule_level_for("초콜릿", "dog", 25).level is TriageLevel.CALL_NOW
