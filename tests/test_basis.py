"""판정 근거 공시 (D-81) — **등급이 어디서 나왔는지 문장에 밝힌다.**

    🔴 지금 전화        ← 50 mg/kg 을 계산해서 나온 3
    🔴 지금 전화        ← 양을 몰라 바닥으로 깔아 둔 3

등급이라는 형식 자체가 확신을 표현하는데, 그 확신의 출처가 응답 어디에도 없었다.
골든셋 `answered` 39건 중 **26건(67%)이 결손 상태에서 등급을 낸다.**

이 파일은 세 가지를 본다 —
  ① 경로마다 근거 이름이 맞게 붙는가
  ② 밝히지 않으면 **응답이 만들어지지 않는가** (계약 강제)
  ③ 공시 문안이 **골든셋 금지 문구를 밟지 않는가** (기존 통과 건을 깨뜨리지 않는다)
"""

from __future__ import annotations

import csv
import glob

import pytest

from pettriage.compute.vocabulary import resolve_substance
from pettriage.graph.engine import _basis_notice, _basis_of
from pettriage.graph.nodes.compute import compute_metrics
from pettriage.graph.nodes.triage import apply_rule_table
from pettriage.triage.basis import MUST_STATE, TAGS, notice, stated_in


def _state(surface: str | None, amount=None, weight=None, species="dog", **kw):
    slots: dict = {"species": species}
    if surface:
        slots["substance"] = resolve_substance(surface, species).name
    if weight is not None:
        slots["weight_kg"] = weight
    if amount is not None:
        slots["amount_g"] = amount
    st: dict = {"slots": slots, **kw}
    st.update(compute_metrics(st))  # type: ignore[arg-type]
    st.update(apply_rule_table(st))  # type: ignore[arg-type]
    return st


# ── ① 경로마다 이름이 맞는가 ────────────────────────────────────
class Test근거이름:
    def test_계산했으면_정량계산(self):
        st = _state("밀크초콜릿", 20.0, 5.0)
        assert _basis_of(st) == "정량계산"
        assert "8 mg/kg" in _basis_notice(st)
        assert "2.0 mg/g" in _basis_notice(st), "무엇으로 계산했는지가 없다"

    def test_양을_모르면_양미상(self):
        """G-023 — *"몇 알인지는 못 봤어요"*. **적게 먹었다는 뜻이 아니다.**"""
        st = _state("다크초콜릿", None, 5.0)
        assert _basis_of(st) == "양미상"
        assert "적게 먹었다는 뜻이 아니라" in _basis_notice(st)

    def test_체중을_모르면_그것을_말한다(self):
        st = _state("다크초콜릿", 50.0, None)
        assert _basis_of(st) == "양미상"
        assert "체중" in _basis_notice(st)

    def test_정성표가_잡으면_정성(self):
        """포도는 정성 등급이 있고 역치 행이 0이다 — 양을 알아도 계산할 것이 없다."""
        st = _state("포도", None, 5.0)
        assert _basis_of(st) == "정성"

    def test_규칙이_아무것도_못_내면_모델판정(self):
        """🔴 **수치 근거가 전혀 없는 등급이다.** 가장 먼저 밝혀야 한다."""
        st = {"slots": {"species": "dog", "substance": "초콜릿"}, "llm_level": 4}
        assert _basis_of(st) == "모델판정"
        assert "수치 근거가 없습니다" in _basis_notice(st)

    def test_이름과_태그가_짝을_이룬다(self):
        for name in MUST_STATE:
            assert name in TAGS
            assert TAGS[name] in notice(name, mg_per_kg=1.0)


# ── ② 밝히지 않으면 응답이 만들어지지 않는가 ────────────────────
class Test계약강제:
    """붙이기로 **약속**하면 언젠가 빠진다. 여기서 막으면 못 빠뜨린다 (D-40)."""

    @staticmethod
    def _resp(answer: str, basis: str | None):
        from pettriage.app.contracts import AskResponse, Citation, TriageResult

        return AskResponse(
            status="answered",
            session_id="s-1",
            answer=answer,
            triage=TriageResult(
                level=3, message="지금 전화", escalation_conditions=["구토"], basis=basis
            ),
            citations=[Citation(source_id="S-034", publisher="x")],
        )

    def test_밝히면_통과한다(self):
        r = self._resp("[확인 안 됨] 섭취량을 확인하지 못했습니다. 지금 전화하세요.", "양미상")
        assert r.triage is not None and r.triage.basis == "양미상"

    @pytest.mark.parametrize("basis", sorted(MUST_STATE))
    def test_안_밝히면_응답을_못_만든다(self, basis):
        with pytest.raises(ValueError, match="판정 근거"):
            self._resp("초콜릿은 개에게 독성이 있습니다.", basis)

    def test_기본값_None_은_검사하지_않는다(self):
        """**기존 경로를 깨지 않는다.** 새 값을 넣는 쪽만 검사를 받는다."""
        r = self._resp("초콜릿은 개에게 독성이 있습니다.", None)
        assert r.triage is not None and r.triage.basis is None

    def test_태그를_계약에_다시_적지_않았다(self):
        """만드는 쪽과 검사하는 쪽이 각자 적으면 어긋난다 (D-22).

        어긋나는 순간 계약이 통과시키고, **강제가 사라진다.**
        """
        import inspect

        from pettriage.app import contracts

        src = inspect.getsource(contracts)
        for tag in TAGS.values():
            assert tag not in src, f"계약에 태그 {tag!r} 가 박혀 있다"

    def test_stated_in_은_태그로_판정한다(self):
        assert stated_in("양미상", "앞말 [확인 안 됨] 뒷말")
        assert not stated_in("양미상", "섭취량을 확인하지 못했습니다")
        assert stated_in(None, "")


# ── ③ 기존 통과 건을 깨뜨리지 않는가 ────────────────────────────
class Test금지문구를안밟는다:
    """🔴 **공시 문장이 `must_not_contain` 을 밟으면 지금 통과하는 건이 실패로 돌아선다.**

    실제로 하나 걸렸다 — `[계산]` 문안의 `mg/kg` 가 G-005·G-013·G-017 의 금지어다.
    셋 다 **조류**이고, 조류는 정량 역치가 0행이라(D-09) `정량계산` 이 될 수 없어
    구조적으로 안 밟는다. 다만 **말로만 안전한 것은 안 되므로** 아래에 박아 둔다.
    """

    @staticmethod
    def _rows() -> list[dict]:
        """⚠️ **하네스와 같은 글롭을 쓴다** (`run_eval.py` 의 `golden_*.csv`).

        처음에는 `*.csv` 로 읽었다. 그랬더니 채점에 안 들어가는 파일까지 읽어
        **검사하는 쪽과 채점하는 쪽이 다른 골든셋을 봤다** — 그런 검사는
        통과해도 아무것도 보장하지 않는다 (2026-08-03, D-22).
        """
        out: list[dict] = []
        for f in glob.glob("eval/goldenset/golden_*.csv"):
            with open(f, encoding="utf-8-sig") as fh:
                out += list(csv.DictReader(fh))
        return out

    @classmethod
    def _forbidden(cls) -> set[str]:
        out: set[str] = set()
        for r in cls._rows():
            out |= {x.strip() for x in (r["must_not_contain"] or "").split("|") if x.strip()}
        return out

    def test_결손_공시는_금지어를_전혀_안_쓴다(self):
        bad = self._forbidden()
        for basis in ("양미상", "최악가정", "정성", "모델판정"):
            text = notice(basis)
            hits = [t for t in bad if t in text]
            assert not hits, f"{basis} 공시가 금지어를 쓴다: {hits}\n{text}"

    def test_조류는_정량계산이_될_수_없다(self):
        """`[계산]` 문안의 `mg/kg` 를 금지한 케이스가 전부 조류다 (D-09).

        조류에 정량 역치가 생기는 날 이 검사가 먼저 깨진다 — 그때 골든셋과 함께 본다.
        """
        st = _state("초콜릿", 5.0, 0.4, species="bird")
        assert _basis_of(st) != "정량계산"
        assert (st.get("computed") or {}).get("active_mg_per_kg") is None

    def test_수치_공시를_금지한_케이스는_정량계산이_불가능한_종이다(self):
        """`[계산]` 문안은 단위 문자열(`mg/kg`)을 쓴다.

        ⚠️ 처음에는 **금지 어휘를 `{"mg/kg"}` 로 박아 뒀다.** 골든셋이 바뀌자
        (`g/kg` 추가) 곧바로 깨졌는데, 정작 확인해야 하는 것은 *어떤 낱말인가*가
        아니라 **그 케이스가 정량계산이 될 수 있는 종인가**다. 어휘를 박으면
        골든셋이 손질될 때마다 무의미하게 깨지고, 진짜 충돌은 못 잡는다.

        조류가 단위를 금지하는 것은 안전하다 — 조류는 정량 역치가 0행이라
        `정량계산` 이 될 수 없다 (D-09). **개·고양이가 금지하면 진짜 충돌이고,
        지금 통과하는 건이 실패로 돌아선다.** 그때 여기서 먼저 걸린다.
        """
        text = notice("정량계산", active="테오브로민", mg_per_kg=8.0, detail="2 mg/g x 20 g")
        offenders = [
            (r["case_id"], r["species"], [t for t in terms if t in text])
            for r in self._rows()
            if (terms := [x.strip() for x in (r["must_not_contain"] or "").split("|") if x.strip()])
            and any(t in text for t in terms)
            and r["species"] != "bird"
        ]
        assert not offenders, (
            "정량계산이 가능한 종의 케이스가 수치 공시의 낱말을 금지한다 — "
            f"공시를 넣으면 이 건들이 실패로 돌아선다: {offenders}"
        )


class Test행동문장:
    """D-89 — **등급이 시킨 일을 문장에도 싣는다.**

    D-81 은 근거에 대해 *"배지를 읽고 문장은 흘린다"* 고 했다. 행동은 방향이
    반대였다 — **배지만 있고 문장이 없었다.** 2026-08-03 실측: 등급
    `CALL_NOW` 이상 36건 중 **19건(53%)** 이 `전화`·`병원` 을 본문에 안 썼다.
    """

    @staticmethod
    def _rows():
        rows = []
        for path in sorted(glob.glob("eval/goldenset/golden_*.csv")):
            with open(path, encoding="utf-8-sig", newline="") as f:
                rows += list(csv.DictReader(f))
        return rows

    def test_등급마다_행동_문장이_있다(self):
        from pettriage.triage.levels import TriageLevel

        for lv in TriageLevel:
            assert TriageLevel(lv).message, f"{lv.name} 에 행동 문장이 없다"

    def test_행동_문장은_등급의_말을_쓴다(self):
        from pettriage.triage.levels import TriageLevel

        assert "병원" in TriageLevel.EMERGENCY.message
        assert "전화" in TriageLevel.CALL_NOW.message
        assert "진료" in TriageLevel.VISIT_SOON.message

    def test_공시는_맨_끝에_붙는다(self):
        """근거는 **단서**라 앞, 행동은 **결론**이라 뒤 (D-81 · D-89)."""
        from pettriage.graph.engine import _advice_line

        line = _advice_line({"triage_level": 3})  # type: ignore[arg-type]
        assert line.startswith("[해야 할 일]")
        assert "전화" in line

    def test_MONITOR는_상승_조건을_함께_낸다(self):
        """*"아래 증상이 나타나면"* 이라고 해 놓고 아래가 비면 안 된다 (D-39)."""
        from pettriage.graph.engine import _advice_line

        line = _advice_line(
            {"triage_level": 1, "escalation_conditions": ["구토", "설사"]}  # type: ignore[arg-type]
        )
        assert "구토" in line and "설사" in line

    def test_등급이_없으면_아무것도_안_붙인다(self):
        from pettriage.graph.engine import _advice_line

        assert _advice_line({}) == ""  # type: ignore[arg-type]
        assert _advice_line({"triage_level": None}) == ""  # type: ignore[arg-type]

    def test_행동_문장이_골든셋_금지_문구를_밟지_않는다(self):
        """🔴 **기존 통과 건을 깨뜨리지 않는다.**

        정답 등급의 행동 문장이 그 케이스의 금지 문구를 밟으면
        골든셋이 자기모순이다 — 그 등급을 내라고 하면서 그 등급의 말을 금지한 것이다.
        """
        from pettriage.triage.levels import TriageLevel

        name_to_level = {lv.name: int(lv) for lv in TriageLevel}
        offenders = []
        for r in self._rows():
            level = name_to_level.get(r["expected_triage"])
            if level is None:
                continue
            msg = TriageLevel(level).message
            hit = [
                t.strip()
                for t in (r["must_not_contain"] or "").split("|")
                if t.strip() and t.strip() in msg
            ]
            if hit:
                offenders.append((r["case_id"], r["expected_triage"], hit))
        assert not offenders, f"행동 문장을 실으면 이 건들이 실패로 돌아선다: {offenders}"
