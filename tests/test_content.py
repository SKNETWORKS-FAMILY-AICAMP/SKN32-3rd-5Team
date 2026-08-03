"""성분 함량 계수 (D-78) — **먹은 무게 ≠ 유효성분 무게.**

    밀크초콜릿 20 g / 5 kg  →  4,000 mg/kg  →  중증(60) 초과  →  EMERGENCY
    테오브로민 2 mg/g 적용  →      8 mg/kg  →  임상징후(20) 미만 → MONITOR

**200배 틀렸고 방향이 과대였다.** 골든셋이 `MONITOR` 라고 적어 둔 질의에
*"지금 병원"* 을 내는 것은 의도된 편향(D-13)이 아니라 오답이다.

이 파일은 두 가지를 본다 —
  ① 손으로 적은 표가 **사실 표와 어긋나지 않는가** (근거 없는 계수 금지, D-16)
  ② 그 계수가 **실제 판정을 바꾸는가** (표만 있고 안 쓰이던 전례가 있다, D-75)
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pettriage.compute.content import (
    BOUNDS,
    ContentTableError,
    content_for,
    load_contents,
    threshold_substance,
)

ROOT = Path(__file__).resolve().parents[1]
FACTS_DIR = ROOT / "data" / "facts"


@pytest.fixture(scope="module")
def facts() -> dict[str, dict[str, str]]:
    """사실 표를 fact_id 로 색인한다. **이것이 계수의 원본이다.**"""
    out: dict[str, dict[str, str]] = {}
    for p in sorted(FACTS_DIR.glob("facts_*.csv")):
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            fid = (r.get("fact_id") or "").strip()
            if fid:
                out[fid] = r
    if not out:
        pytest.skip("사실 표가 없다 (data/facts/)")
    return out


# ── ① 표가 자료와 어긋나지 않는가 ───────────────────────────────
class Test표는자료를따른다:
    def test_모든_행에_근거가_있다(self):
        """**근거 없는 계수는 우리가 만든 숫자가 등급을 판정하게 둔다** (D-16)."""
        for c in load_contents():
            assert c.basis, c.substance
            assert c.bound in BOUNDS

    def test_수치가_사실_표와_같다(self, facts):
        """손으로 옮긴 값이 원본과 같은가.

        `별칭.csv` 가 `basis` 를 대조받는 것과 같은 이유다 — 손으로 적은 표는
        **적은 순간이 아니라 자료가 바뀐 뒤에** 틀린다.
        """
        for c in load_contents():
            for fid in c.basis:
                assert fid in facts, f"{c.substance}: 없는 fact_id {fid}"
                row = facts[fid]
                assert (
                    row["substance"] == c.substance
                ), f"{fid} 의 물질은 {row['substance']!r} 인데 표는 {c.substance!r} 로 적었다"
                assert (
                    (row.get("threshold_type") or "").strip() == "성분 함량"
                ), f"{fid} 는 '성분 함량' 행이 아니다: {row.get('threshold_type')!r}"
                dose = (row.get("dose") or "").strip()
                if c.mg_per_g is None:
                    assert not dose, f"{fid} 에 수치가 있는데 표는 비워 뒀다: {dose!r}"
                else:
                    assert dose, f"{fid} 에 수치가 없는데 표는 {c.mg_per_g} 로 적었다"
                    digits = "".join(ch for ch in dose if ch.isdigit() or ch == ".")
                    assert float(digits) == c.mg_per_g, f"{fid}: {dose!r} ≠ {c.mg_per_g}"
                    # 부등호가 있으면 **하한**으로 표시돼 있어야 한다.
                    assert (
                        ">" in dose or "≥" in dose
                    ) == c.is_lower_bound, f"{fid}: 원문 {dose!r} 과 bound={c.bound!r} 가 어긋난다"

    def test_역치물질에_실제로_역치가_있다(self):
        """🔴 **이 연결이 끊기면 판정불가 거절이 된다.**

        `밀크 초콜릿` 은 어휘표에 있지만 역치 행이 0개다. 물질을 정확히 알아낸
        질의가 마지막에서 버려지는 것이 이 검사가 막는 사고다.
        """
        from pettriage.compute.rules import lookup

        for c in load_contents():
            for sp in c.species or ("dog",):
                assert lookup(
                    c.active_substance, sp
                ), f"{c.substance}({sp}) → {c.active_substance} 에 역치 행이 없다"

    def test_물질이_어휘표에_있다(self):
        """② 슬롯이 올릴 수 없는 이름이면 이 행은 **영원히 안 걸린다.**"""
        from pettriage.compute.vocabulary import resolve_substance

        for c in load_contents():
            assert resolve_substance(c.substance, None).name == c.substance, c.substance


# ── ② 계수가 판정을 바꾸는가 ────────────────────────────────────
def _rule_level(surface: str, amount_g: float, weight_kg: float, species: str = "dog"):
    from pettriage.compute.vocabulary import resolve_substance
    from pettriage.graph.nodes.compute import compute_metrics
    from pettriage.graph.nodes.triage import apply_rule_table

    name = resolve_substance(surface, species).name
    st: dict = {
        "slots": {
            "species": species,
            "substance": name,
            "weight_kg": weight_kg,
            "amount_g": amount_g,
        }
    }
    st.update(compute_metrics(st))  # type: ignore[arg-type]
    st.update(apply_rule_table(st))  # type: ignore[arg-type]
    return st.get("rule_level"), st.get("computed", {})


class Test판정이바뀐다:
    @pytest.mark.parametrize(
        ("case", "surface", "amount_g", "expected"),
        [
            ("G-028", "밀크초콜릿", 20.0, 1),  # 8 mg/kg  → 역치 미만
            ("G-029", "밀크초콜릿", 60.0, 3),  # 24 mg/kg → 임상징후(20)
            ("G-030", "다크초콜릿", 50.0, 4),  # 50 mg/kg → 중증(40-50)
            ("G-041", "화이트초콜릿", 100.0, 1),  # 수치 없음 → 역치 미만
        ],
    )
    def test_골든셋_등급에_떨어진다(self, case, surface, amount_g, expected):
        """네 건이 **골든셋이 적어 둔 등급에 정확히** 떨어진다.

        골든셋을 이 사실들(F-034-020~026)에서 뽑았으니 당연하지만, 그래서 검산이 된다.
        계수를 빼면 넷 다 EMERGENCY 로 뭉개진다.
        """
        level, _ = _rule_level(surface, amount_g, 5.0)
        assert level == expected, f"{case} {surface} {amount_g}g"

    def test_계수를_안_쓰면_뭉개진다(self):
        """**이 검사가 회귀를 잡는다.** 계수를 안 태우면 20g 과 100g 이 같은 등급이 된다."""
        from pettriage.compute.rules import rule_level_for, to_mg_per_kg

        raw = to_mg_per_kg(20.0 / 5.0, "g/kg")  # 예전 계산 — 물질 무게 그대로
        assert raw == 4000.0
        assert int(rule_level_for("초콜릿(테오브로민+카페인)", "dog", raw).level) == 4

    def test_수치를_0으로_지어내지_않는다(self):
        """화이트 초콜릿은 원문이 *'무의미한 수준'* 이라고만 한다 — **숫자가 아니다.**

        0 을 적으면 원문이 한 적 없는 강도로 *"안전하다"* 를 말하게 된다.
        정량 판정을 건너뛰고 역치 미만 경로로 보낸다 (답은 같지만 근거가 다르다).
        """
        c = content_for("화이트 초콜릿", "dog")
        assert c is not None
        assert c.mg_per_g is None
        assert not c.quantifiable

        _, computed = _rule_level("화이트초콜릿", 100.0, 5.0)
        assert "active_mg_per_kg" not in computed, "정량 판정을 했다"

    def test_계수가_응답_수치에_남는다(self):
        """**계산했다는 증거가 응답에 있어야 한다** (D-75 · 산출물 ④)."""
        _, computed = _rule_level("다크초콜릿", 50.0, 5.0)
        assert computed["active_substance"] == "초콜릿(테오브로민+카페인)"
        assert computed["content_mg_per_g"] == 5
        assert computed["active_mg"] == 250.0
        assert computed["active_mg_per_kg"] == 50.0
        assert "F-034-024" in computed["content_basis"]


class Test계수없는물질:
    def test_이름이_그대로_돌아온다(self):
        """계수가 없는 물질은 **예전 그대로** 돈다 — 마늘 역치는 물질 무게 기준이다."""
        assert threshold_substance("마늘", "dog") == "마늘"
        assert content_for("마늘", "dog") is None

    def test_마늘은_바뀌지_않았다(self):
        """회귀 확인 — 30g/5kg = 6 g/kg 은 마늘 역치 5 g/kg 초과다."""
        level, computed = _rule_level("다진마늘", 30.0, 5.0)
        assert level == 3
        assert computed["dose_per_kg"] == 6.0
        assert "active_mg" not in computed


class Test표가규칙을어기면:
    def test_bound_가_이상한_행은_전체를_세운다(self, tmp_path, monkeypatch):
        """**조용히 넘어가지 않는다** — 계수가 빠지면 200배 틀린다."""
        import pettriage.compute.content as mod

        bad = tmp_path / "성분함량.csv"
        bad.write_text(
            "substance,species,active_substance,mg_per_g,bound,basis,source_id,note\n"
            "가짜,dog,초콜릿(테오브로민+카페인),5,대충,F-000-000,S-000,\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "_table_path", lambda: bad)
        mod.load_contents.cache_clear()
        with pytest.raises(ContentTableError, match="bound"):
            mod.load_contents()
        mod.load_contents.cache_clear()

    def test_근거_없는_행은_전체를_세운다(self, tmp_path, monkeypatch):
        import pettriage.compute.content as mod

        bad = tmp_path / "성분함량.csv"
        bad.write_text(
            "substance,species,active_substance,mg_per_g,bound,basis,source_id,note\n"
            "가짜,dog,초콜릿(테오브로민+카페인),5,정확,,S-000,\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(mod, "_table_path", lambda: bad)
        mod.load_contents.cache_clear()
        with pytest.raises(ContentTableError, match="basis"):
            mod.load_contents()
        mod.load_contents.cache_clear()


class Test별칭:
    @pytest.mark.parametrize(
        ("surface", "expected"),
        [
            ("다크초콜릿", "세미스위트 다크 초콜릿"),
            ("밀크초콜릿", "밀크 초콜릿"),
            ("화이트초콜릿", "화이트 초콜릿"),
            ("베이킹초콜릿", "무가당 베이킹 초콜릿·코코아 파우더"),
            ("코코아파우더", "무가당 베이킹 초콜릿·코코아 파우더"),
        ],
    )
    def test_붙여쓴_표기가_올라간다(self, surface, expected):
        """어휘표는 띄어 쓰고(`밀크 초콜릿`) 사람은 붙여 쓴다(`밀크초콜릿`).

        이 다섯 글자 차이 때문에 프로브 8건 중 5건이 되묻기로 빠졌다 (2026-08-02).
        """
        from pettriage.compute.vocabulary import resolve_substance

        r = resolve_substance(surface, "dog")
        assert r.name == expected
        assert not r.assumption, "확정 별칭인데 추정으로 올라왔다"
