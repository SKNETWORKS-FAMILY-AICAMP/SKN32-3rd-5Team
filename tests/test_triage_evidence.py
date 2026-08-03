"""⑨ 트리아지 판정에 **코드가 계산한 값을 준다** (D-79).

2026-08-02 프로브 실측 — 과대평가 4건이 전부 한 경로였다.

    G-028  밀크초콜릿 20 g / 5 kg
           코드: 테오브로민 8 mg/kg (임상징후 역치 20 미만) → MONITOR
           LLM : EMERGENCY
           게이트 max(rule, llm)                            → 최종 4
           골든셋 기대                                       → MONITOR

판정 노드에 넘어가던 것은 **검색 문맥뿐**이었다. 8 mg/kg 인지 4,000 mg/kg 인지
모른 채 `개 + 초콜릿` 만 보고 답했고, 그 상황에서는 그것이 합리적인 답이다.
**눈을 가리고 시킨 판단을 판단이라고 부르지 않는다.**

반대 방향도 하나 있었다 — G-001(`다크초콜릿을 새끼손톱만큼`)은 양을 몰라
`역치 미만`(MONITOR)으로 깔렸다. **모르는 것을 안전으로 읽은 것**이고 과소평가다.
"""

from __future__ import annotations

import pytest

from pettriage.compute.vocabulary import resolve_substance
from pettriage.graph.nodes.compute import compute_metrics, numeric_evidence
from pettriage.graph.nodes.triage import apply_rule_table
from pettriage.triage.levels import TriageLevel


def _state(surface: str, amount_g: float | None, weight_kg: float | None = 5.0, species="dog"):
    slots: dict = {"species": species, "substance": resolve_substance(surface, species).name}
    if weight_kg is not None:
        slots["weight_kg"] = weight_kg
    if amount_g is not None:
        slots["amount_g"] = amount_g
    st: dict = {"slots": slots, "context": "(검색된 근거 문장)"}
    st.update(compute_metrics(st))  # type: ignore[arg-type]
    st.update(apply_rule_table(st))  # type: ignore[arg-type]
    return st


# ── 무엇을 주는가 ───────────────────────────────────────────────
class Test증거블록:
    def test_계산한_수치와_역치를_함께_준다(self):
        ev = numeric_evidence(_state("밀크초콜릿", 20.0))
        assert "8 mg/kg" in ev, ev
        assert "임상징후 발현 20 mg/kg" in ev
        assert "F-034-020" in ev, "역치의 출처가 빠졌다"

    def test_역치는_표에서_온다(self, monkeypatch):
        """프롬프트에 상수로 박으면 **표가 바뀔 때 조용히 어긋난다** (D-22).

        표를 갈아 끼워 그 값이 그대로 나오는지 본다 — 주석에 예시 숫자가 적혀 있어도
        걸리지 않고, *어디서 왔는가* 만 본다.
        """
        from pettriage.compute.rules import Rule
        from pettriage.graph.nodes import compute as compute_mod

        fake = Rule(
            fact_id="F-999-001",
            substance="가짜",
            species="dog",
            threshold_type="임상징후 발현",
            dose="777",
            unit="mg/kg",
            computable=True,
            effect="",
            signs="구토",
            onset="",
            source_id="S-999",
            citation="",
            note="",
        )
        monkeypatch.setattr(compute_mod, "lookup", lambda *a, **k: [fake], raising=False)
        monkeypatch.setattr("pettriage.compute.rules.lookup", lambda *a, **k: [fake])

        ev = numeric_evidence(_state("밀크초콜릿", 20.0))
        assert "777 mg/kg" in ev, ev
        assert "F-999-001" in ev

    def test_양을_모르면_모른다고_적는다(self):
        """**모른다는 사실도 근거다** (D-13)."""
        ev = numeric_evidence(_state("다크초콜릿", None))
        assert "[확인 안 된 것]" in ev
        assert "섭취량" in ev
        assert "안전으로 읽지 않는다" in ev
        assert "[코드가 계산한 값]" not in ev, "못 한 계산을 했다고 적었다"


# ── 무엇을 주지 않는가 ──────────────────────────────────────────
class Test등급은주지않는다:
    """🔴 **규칙의 결론을 주면 `llm_level` 이 독립적이지 않게 된다.**

    LLM 이 규칙을 따라 읽으면 `overridden` 이 영원히 거짓이 되고, 산출물 ④에서
    *"하향 금지 게이트가 실제로 작동했다"* 를 보일 증거가 사라진다 (D-09).
    주는 것은 수치와 역치뿐이고, 판단은 LLM 이 그대로 한다.
    """

    @pytest.mark.parametrize(
        ("surface", "amount_g"), [("밀크초콜릿", 20.0), ("다크초콜릿", 50.0), ("다크초콜릿", None)]
    )
    def test_등급_이름이_안_들어간다(self, surface, amount_g):
        st = _state(surface, amount_g)
        ev = numeric_evidence(st)
        assert st.get("rule_level") is not None, "이 케이스는 규칙이 등급을 냈어야 한다"
        for name in ("MONITOR", "VISIT_SOON", "CALL_NOW", "EMERGENCY"):
            assert name not in ev, f"규칙의 결론 {name!r} 이 새어 나갔다:\n{ev}"
        assert "rule_level" not in ev


# ── 판정 노드가 실제로 받는가 ───────────────────────────────────
class Test판정노드:
    def test_증거가_프롬프트에_실린다(self, monkeypatch):
        """표만 만들고 안 쓰이던 전례가 있다 (D-75). **받는 쪽까지 본다.**"""
        from pettriage.graph.nodes import generate

        seen: dict = {}

        def fake(system, user_input, max_tokens):
            seen["system"], seen["user"] = system, user_input
            return "MONITOR"

        monkeypatch.setattr(generate, "_call_raw", fake)
        out = generate.judge_triage(_state("밀크초콜릿", 20.0))

        assert out == {"llm_level": int(TriageLevel.MONITOR)}
        assert "8 mg/kg" in seen["user"], seen["user"]
        assert "[검색된 근거]" in seen["user"]
        assert "다시 계산하지 않는다" in seen["system"]

    def test_줄_것이_없으면_문맥만_보낸다(self, monkeypatch):
        """물질도 종도 없으면 덧붙일 것이 없다. **빈 머리말을 붙이지 않는다.**"""
        from pettriage.graph.nodes import generate

        seen: dict = {}
        monkeypatch.setattr(
            generate,
            "_call_raw",
            lambda s, u, max_tokens: seen.update(user=u) or "CALL_NOW",
        )
        generate.judge_triage({"context": "근거만 있다"})
        assert seen["user"] == "근거만 있다"


# ── 양 미상의 바닥 ──────────────────────────────────────────────
class Test양미상바닥:
    def test_양을_모르면_전화_바닥(self):
        """G-001 — *"다크초콜릿을 새끼손톱만큼"*.

        역치가 있다는 것은 **이 종에 이 물질의 독성이 확인됐다**는 뜻이다.
        거기에 양을 모른다면 할 일은 관찰이 아니라 **지금 전화해서 물어보는 것**이다.
        """
        st = _state("다크초콜릿", None)
        assert st["rule_level"] == int(TriageLevel.CALL_NOW)
        assert st.get("escalation_conditions"), "바닥만 올리고 조건을 안 줬다"

    def test_체중을_몰라도_같다(self):
        st = _state("다크초콜릿", 50.0, weight_kg=None)
        assert st["rule_level"] == int(TriageLevel.CALL_NOW)

    def test_재고_나서_역치_미만이면_관찰_그대로(self):
        """**바꾼 것은 '못 쟀다' 쪽뿐이다.** 재서 낮게 나온 것은 낮은 것이 맞다."""
        st = _state("밀크초콜릿", 20.0)
        assert st["rule_level"] == int(TriageLevel.MONITOR)
        assert st["computed"]["active_mg_per_kg"] == 8.0

    def test_역치가_없으면_바닥을_만들지_않는다(self):
        """근거가 없으면 **없다고 한다** — `decide_triage` 가 판정불가로 보낸다 (D-10).

        양 미상 바닥이 *"모르면 일단 3"* 이 되어선 안 된다. 역치 행의 존재가 근거다.
        """
        st: dict = {"slots": {"species": "cat", "substance": "밀크 초콜릿", "weight_kg": 4.0}}
        assert apply_rule_table(st) == {}  # type: ignore[arg-type]


# ── 단위를 맞춰서 낸다 ──────────────────────────────────────────
class Test단위:
    """🔴 2026-08-02 실측 — G-041 에서 **단위가 다른 두 숫자를 나란히 냈다.**

        [코드가 계산한 값]  화이트 초콜릿 20 g/kg      ← 초콜릿 무게
        [자료의 역치]      임상징후 발현 20 mg/kg     ← 테오브로민 기준

    1000배 넘긴 것처럼 보인다. 그 정보만 보면 EMERGENCY 가 합리적인 답이고
    실제로 그렇게 나왔다. **오도한 것은 모델이 아니라 우리다.**
    """

    @staticmethod
    def _computed_line(ev: str) -> str:
        lines = ev.splitlines()
        return lines[lines.index("[코드가 계산한 값]") + 1]

    def test_계산값은_항상_mg_per_kg(self):
        """마늘 역치는 원문이 `5 g/kg` 이다. 계산값을 `6 g/kg` 로 내면 자릿수가 같아 보이고,
        `6000 mg/kg` 로 내면 `5000 mg/kg` 와 바로 견줄 수 있다."""
        line = self._computed_line(numeric_evidence(_state("다진마늘", 30.0)))
        assert "6000 mg/kg" in line, line
        assert " g/kg" not in line, "물질 무게를 g/kg 로 내면 mg/kg 역치와 단위가 안 맞는다"

    def test_역치도_같은_단위로_환산해_준다(self):
        """원문 표기는 남기되(`5 g/kg`) 환산값을 함께 적는다 — 읽는 쪽이 맞추게 두지 않는다."""
        ev = numeric_evidence(_state("다진마늘", 30.0))
        assert "5 g/kg = 5000 mg/kg" in ev, ev

    def test_이미_mg_per_kg_면_중복해_적지_않는다(self):
        ev = numeric_evidence(_state("밀크초콜릿", 20.0))
        assert "20 mg/kg = " not in ev, ev

    def test_함량_무의미는_모름이_아니라_사실이다(self):
        """**없음(정보 부재)과 무의미함(확인된 사실)을 같은 칸에 두지 않는다.**

        같이 두면 뒤에 붙는 *"모르는 것을 안전으로 읽지 않는다"* 가
        **자료가 말한 것까지 밀어 올린다.**
        """
        ev = numeric_evidence(_state("화이트초콜릿", 100.0))
        assert "[자료가 말하는 것]" in ev
        assert "무의미한 수준" in ev
        assert "F-034-026" in ev
        assert "[확인 안 된 것]" not in ev, ev
        assert "[코드가 계산한 값]" not in ev, "수치를 못 내는데 냈다"


# ── 무의미함은 모름이 아니다 (판정 쪽) ──────────────────────────
class Test무의미와모름:
    """🔴 2026-08-02 G-041 실측 — 프롬프트에서는 갈라 놓고 **판정 코드는 안 갈랐다.**

        "강아지가 화이트초콜릿 100g을 먹었는데"   ← 체중이 문장에 없다
        rule = CALL_NOW  (양 미상 바닥, D-79)
        llm  = MONITOR   (자료를 읽고 제대로 답했다)
        게이트가 LLM 을 막았다 → 최종 3 · 골든셋 기대 1

    바닥의 논리는 *"독성이 확인된 물질 + 양 미상 → 전화해서 확인"* 이다.
    **유효성분 함량 자체가 무의미하면 그 전제가 성립하지 않는다** —
    물어볼 이유가 없는 것을 모른다고 등급을 올리면 되묻기가 아니라 **겁주기**다.
    """

    def test_체중을_몰라도_무의미면_관찰(self):
        st = _state("화이트초콜릿", 100.0, weight_kg=None)
        assert st["rule_level"] == int(TriageLevel.MONITOR)

    def test_체중을_알아도_같다(self):
        st = _state("화이트초콜릿", 100.0, weight_kg=5.0)
        assert st["rule_level"] == int(TriageLevel.MONITOR)

    def test_무의미가_아니면_바닥은_그대로_전화(self):
        """**면제는 '자료가 무의미하다고 말한' 물질에만 준다.** 넓히면 D-79 가 무력해진다."""
        assert _state("다크초콜릿", None)["rule_level"] == int(TriageLevel.CALL_NOW)
        assert _state("다진마늘", 30.0, weight_kg=None)["rule_level"] == int(TriageLevel.CALL_NOW)

    def test_무의미와_계산가능은_다른_질문이다(self):
        """`quantifiable` 은 *계산할 수 있나*, `negligible` 은 *왜 못 하나* 다."""
        from pettriage.compute.content import content_for

        white = content_for("화이트 초콜릿", "dog")
        milk = content_for("밀크 초콜릿", "dog")
        assert white is not None and milk is not None
        assert white.negligible and not white.quantifiable
        assert not milk.negligible and milk.quantifiable


# ── D-80 · 잰 자리에서는 규칙이, 못 잰 자리에서는 LLM 이 ─────────
class Test정량이면상승을막는다:
    """2026-08-03 골든셋 60건 (`--arm A` · gpt-4o-mini) —

        LLM 이 올렸다  10건   그중 **과대로 끝난 것 7건** (70%)
        틀린 상승 7건 중 6건이 `rule=3 llm=4` · 유형은 `dose` 가 최악 (과대 42.9%)

        반대로 옳았던 상승도 있다 — 기준선에서 과소였던 G-011(백합·cat)·
        G-017(아보카도·bird)이 LLM 상승으로 고쳐졌다. **둘 다 못 잰 자리**다.

    경계가 선명하다. **하향 금지는 그대로 두고 상향에만 조건을 단다** (D-09 개정).
    """

    def test_잰_자리에서는_못_올린다(self):
        from pettriage.triage.gate import apply_gate

        d = apply_gate(rule_level=3, llm_level=4, rule_is_quantitative=True)
        assert int(d.level) == 3
        assert d.llm_capped
        assert int(d.llm_level) == 4, "막았다고 기록까지 지우면 안 된다"

    def test_못_잰_자리에서는_올린다(self):
        """G-011·G-017 이 이 경로로 고쳐졌다. 여기까지 잠그면 과소가 되돌아온다."""
        from pettriage.triage.gate import apply_gate

        d = apply_gate(rule_level=3, llm_level=4, rule_is_quantitative=False)
        assert int(d.level) == 4
        assert not d.llm_capped

    def test_하향은_여전히_막는다(self):
        """**개정한 것은 상향 조건뿐이다.** D-09 의 본체는 그대로다."""
        from pettriage.triage.gate import apply_gate

        for quant in (True, False):
            d = apply_gate(
                rule_level=3,
                llm_level=1,
                escalation_conditions=("구토",),
                rule_is_quantitative=quant,
            )
            assert int(d.level) == 3
            assert d.overridden
            assert not d.llm_capped

    def test_정량_표시는_계산_노드만_세운다(self):
        """정성 표·양 미상 바닥은 세우지 않는다 — 거기서는 LLM 이 맞는 경우가 있다."""
        from pettriage.graph.nodes.compute import compute_metrics
        from pettriage.graph.nodes.triage import apply_rule_table

        quant = compute_metrics(
            {
                "slots": {
                    "species": "dog",
                    "substance": "밀크 초콜릿",
                    "weight_kg": 5.0,
                    "amount_g": 20.0,
                }
            }
        )
        assert quant.get("rule_is_quantitative") is True

        floor = apply_rule_table(
            {"slots": {"species": "dog", "substance": "세미스위트 다크 초콜릿", "weight_kg": 5.0}}
        )
        assert floor.get("rule_level") is not None
        assert "rule_is_quantitative" not in floor, "양 미상 바닥을 잰 것으로 표시했다"

    def test_그래프가_게이트에_전달한다(self):
        """플래그를 만들어 놓고 안 넘기던 전례가 있다 (D-75). **끝까지 본다.**"""
        from pettriage.graph.nodes.triage import decide_triage

        out = decide_triage(
            {"rule_level": 3, "llm_level": 4, "rule_is_quantitative": True}  # type: ignore[arg-type]
        )
        assert out["triage_level"] == 3
        assert out["llm_capped"] is True

    def test_계약이_막힌_등급을_받는다(self):
        """`_level_is_final` 은 `level >= max(rule, llm)` 을 요구했다.

        막으면 `level < llm_level` 이 되므로 **계약이 먼저 터진다** — 개정을 코드 한
        곳에만 넣으면 응답을 만들 수 없다 (D-40 · D-72).
        """
        from pettriage.app.contracts import TriageResult

        t = TriageResult(level=3, message="지금 전화", rule_level=3, llm_level=4, llm_capped=True)
        assert t.level == 3 and t.llm_capped

    def test_막았다고_적고_안_막으면_거부한다(self):
        """감사 정보가 거짓이면 산출물 ④가 거짓이 된다."""
        from pettriage.app.contracts import TriageResult

        with pytest.raises(ValueError, match="llm_capped"):
            TriageResult(level=4, message="지금 병원", rule_level=3, llm_level=4, llm_capped=True)


# ── D-84 · 조건 없는 MONITOR 를 침묵으로 바꾸지 않는다 ────────────────
class Test조건없는MONITOR:
    """🔴 **거절은 과소평가를 침묵으로 바꿀 뿐이다** (D-84 · 2026-08-03 실측).

    D-39 는 *"조건 없는 관찰은 그 자체가 과소평가"* 라고 옳게 판정했고, 그에 대한
    조치로 출력을 막았다. 60건 측정에서 그 조치가 무엇을 만드는지가 드러났다 —

        G-012 살충제 · G-018 아스피린 · G-046 고양이 초콜릿
        셋 다 정답 CALL_NOW 인데 **사용자가 아무것도 받지 못했다.**

    셋 다 사실 표에는 있는데 규칙 표에 등급도 역치도 없어(`triage_level`·
    `threshold_type` 공란) `rule_level` 이 서지 않았고, 바닥이 없으니 LLM 의
    MONITOR 가 그대로 최종이 됐다. **바닥이 없는 자리에서 관찰로 끝낼 수는 없다.**
    """

    @staticmethod
    def _hit(doc_type: str):
        class _Chunk:
            def __init__(self, dt):
                self.doc_type = dt

        class _Hit:
            def __init__(self, dt):
                self.chunk = _Chunk(dt)

        return _Hit(doc_type)

    def test_독성_근거면_전화로_올린다(self):
        from pettriage.graph.nodes.triage import decide_triage

        st = {
            "rule_level": None,
            "llm_level": int(TriageLevel.MONITOR),
            "escalation_conditions": [],
            "hits": [self._hit("emergency")],
            "intent": "intoxication",
        }
        out = decide_triage(st)  # type: ignore[arg-type]
        assert out.get("status") != "refused", "독성 근거인데 침묵으로 끝났다 (D-84)"
        assert out["triage_level"] == int(TriageLevel.CALL_NOW)
        assert out["rule_basis"] == "조건미비", "왜 올렸는지 밝히지 않으면 배지만 남는다 (D-81)"

    def test_영양_근거면_올리지_않는다(self):
        """블루베리(G-007)까지 올리면 되묻기가 아니라 **겁주기**다 (D-79 · G-041 교훈)."""
        from pettriage.graph.nodes.triage import decide_triage

        st = {
            "rule_level": None,
            "llm_level": int(TriageLevel.MONITOR),
            "escalation_conditions": [],
            "hits": [self._hit("nutrition")],
            "intent": "nutrition",
        }
        out = decide_triage(st)  # type: ignore[arg-type]
        assert out.get("status") == "refused"
        assert out.get("monitor_without_conditions") is True, "채점이 과소평가로 세야 한다"

    def test_상승_조건이_있으면_그대로_관찰이다(self):
        from pettriage.graph.nodes.triage import decide_triage

        st = {
            "rule_level": int(TriageLevel.MONITOR),
            "llm_level": None,
            "escalation_conditions": ["구토"],
            "hits": [self._hit("emergency")],
            "intent": "intoxication",
        }
        out = decide_triage(st)  # type: ignore[arg-type]
        assert out["triage_level"] == int(TriageLevel.MONITOR)

    def test_급여_질의는_독성_문서가_섞여도_올리지_않는다(self):
        """G-007 블루베리 — 검색이 초콜릿 자료를 딸려왔다고 전화를 걸게 하지 않는다.

        **무엇을 물었는지는 히트가 아니라 ①분류가 안다** (2026-08-03 실측).
        """
        from pettriage.graph.nodes.triage import decide_triage

        st = {
            "rule_level": None,
            "llm_level": int(TriageLevel.MONITOR),
            "escalation_conditions": [],
            "hits": [self._hit("nutrition"), self._hit("toxicity_food")],
            "intent": "nutrition",
        }
        out = decide_triage(st)  # type: ignore[arg-type]
        assert out.get("status") == "refused", "급여 질의를 독성 히트만 보고 올렸다 (D-84)"
