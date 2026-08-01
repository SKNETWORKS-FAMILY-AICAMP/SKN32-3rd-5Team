"""규칙 테이블 조회 — **계산 노드가 수치를 찾는 유일한 경로.**

설계 근거: docs/06_설계결정기록.md · D-16 · D-17 · D-22 · D-39 · D-46

    수치는 벡터 검색으로 찾지 않는다 (D-16). 표를 조회하고 계산은 코드가 한다.

이 모듈이 존재하는 이유
--------------------
`정량임계치.csv` 는 `scripts/build_rule_table.py` 가 사실 표에서 뽑아내는 **생성물**이다.
그런데 그 표를 **어떻게 읽어야 하는가**가 README 산문에만 있으면
읽는 쪽이 지나칠 수 있다. 실제로 검수에서 나온 사고가 정확히 그 종류였다.

    `F-030-010` 주목의 단위는 **`g leaves/kg`** 다 — *잎* 기준이다.
    이것을 `g/kg` 로 읽으면 식물 전체 무게로 오독한다.

그래서 **읽는 규칙을 코드로 고정한다.** 표를 직접 `csv.reader` 로 여는 코드를 쓰지 말고
여기를 통한다.

지켜지는 것 네 가지
-----------------
1. **`computable=N` 은 절대 계산에 쓰이지 않는다.** `computable_for()` 가 걸러낸다
2. **범위·부등호는 안전한 쪽(낮은 값)으로 읽는다** — `40-50` → 40, `≥1` → 1
3. **종은 넓혀서 본다** — `dog` 질의는 `dog`·`mammal`·`all` 을 함께 본다 (D-39)
4. **중복 출처는 접는다** — 양파가 S-034·S-098 에 같은 값으로 두 번 있다.
   접지 않으면 같은 근거를 두 번 세게 된다

없으면 없다고 말한다
-----------------
해당 (물질 × 종) 행이 없으면 **빈 리스트**를 돌려준다. 지어내지 않는다.
**조류는 이 표에 한 행도 없다** (D-09) — 조류 정량 질의는 여기서 반드시 빈 결과가 나온다.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

from ..triage.levels import TriageLevel

TABLE_NAME = "정량임계치.csv"

#: 종 질의를 넓히는 규칙. `mammal` 은 개·고양이 공통 값, `all` 은 종 무관이다.
SPECIES_WIDEN: dict[str, tuple[str, ...]] = {
    "dog": ("dog", "mammal", "all"),
    "cat": ("cat", "mammal", "all"),
    "bird": ("bird", "all"),
}

#: 심각도 순서. 같은 물질에 여러 역치가 있으면 낮은 것부터 넘는다.
SEVERITY: dict[str, int] = {"임상징후 발현": 1, "중증": 2, "치사": 3}

#: **역치를 넘겼을 때의 바닥 등급** (D-50).
#:
#: `rule_level` 은 정밀한 판정이 아니라 **바닥**이다 — `final = max(rule, llm)` 이므로
#: LLM 은 올릴 수만 있다 (D-09). 그래서 고를 때 물어야 할 것은
#: *"이게 의학적으로 맞나"* 가 아니라 **"틀렸을 때 어느 쪽으로 틀리나"** 다.
#:
#: 계산 가능한 12행 중 **9행(75%)이 `임상징후 발현`** 이다 — 사실상 이것이 기본 등급이고,
#: 여기를 `CALL_NOW` 로 두어 **LLM 상향 여지를 남긴다.** D-09 게이트의 증거가 여기서 나온다.
#:
#: `중증` 을 `EMERGENCY` 로 올린 이유는 값을 보면 분명하다 —
#: 초콜릿 40-50 mg/kg 은 4kg 개가 다크초콜릿 20g 정도로 경련·부정맥 구간이고,
#: 자일리톨 0.5 g/kg 은 간부전 구간이다. *"지금 전화"* 로 답하면 그것이 과소평가다.
THRESHOLD_TO_LEVEL: dict[str, TriageLevel] = {
    "임상징후 발현": TriageLevel.CALL_NOW,
    "중증": TriageLevel.EMERGENCY,
    "치사": TriageLevel.EMERGENCY,
}

#: 역치를 **안 넘겼을 때**의 바닥. 상승 조건(`signs`)이 반드시 함께 나간다 (D-39).
#:
#: `None` 으로 두지 않는 이유 — *"조금 먹었는데요"* 가 가장 흔한 질의인데,
#: `rule_level=None` 이면 LLM 이 실패하는 순간 `apply_gate` 가 `ValueError` 로 거절한다.
#: **가장 흔한 질문이 가장 잘 깨지는 설계**가 된다.
BELOW_THRESHOLD_LEVEL: TriageLevel = TriageLevel.MONITOR

#: 질량 단위를 `mg/kg` 으로 환산한다. `%` 는 **체중 대비 백분율**이라
#: 1% = 10 g/kg = 10,000 mg/kg 이다.
_MG_PER_KG: dict[str, float] = {"mg/kg": 1.0, "g/kg": 1000.0, "%": 10_000.0}

#: 같은 (물질 × 종 × 역치종류)에서 최대/최소가 이 배수 이상 벌어지면
#: **정량 판정을 포기한다** (D-50).
#:
#: S-034 는 건포도를 본문에서 `2.8 mg/kg`, 같은 논문 Table 1 에서 `2.8-36.4 g/kg` 로 적는다 —
#: **1,000배 차이**다. 낮은 쪽을 바닥으로 쓰는 원칙을 그대로 두면
#: 단위 오류가 섞인 순간 **거의 모든 섭취가 역치 초과**가 된다.
#:
#: 10배는 **단위 오류(1,000배)와 반올림 차이(1.2배)를 가르는 선**이다.
#: 포기는 실패가 아니다 — 정성 답변으로 내려가고, 그 사실이 로그에 남는다 (D-46).
CONFLICT_RATIO: float = 10.0


class RuleTableMissingError(RuntimeError):
    """표를 못 찾았다. **조용히 빈 표로 넘어가지 않는다.**

    빈 표로 돌면 모든 정량 질의가 "근거 없음"이 되어
    *"우리 시스템은 신중하다"* 로 잘못 읽힌다 (04 §8).
    """


@dataclass(frozen=True)
class Rule:
    """규칙 한 행. **`dose` 원문 문자열을 버리지 않는다** — 답변에 그대로 인용한다."""

    fact_id: str
    substance: str
    species: str
    threshold_type: str
    dose: str
    unit: str
    computable: bool
    effect: str
    #: `|` 로 이어 붙인 증상. **역치 미만일 때 MONITOR 의 상승 조건**이 된다 (D-39 · D-50).
    signs: str
    onset: str
    source_id: str
    citation: str
    note: str

    @property
    def low(self) -> float | None:
        """계산에 쓸 값. **범위와 부등호는 낮은 쪽으로 읽는다.**

        `40-50` → 40.0 · `≥1` → 1.0 · `2-2.5` → 2.0

        높은 쪽을 쓰면 40 mg/kg 을 먹은 개가 "아직 안전"으로 나온다.
        """
        return parse_low(self.dose)

    @property
    def severity(self) -> int:
        return SEVERITY.get(self.threshold_type, 0)

    def key(self) -> tuple[str, str, str, str, str]:
        """출처만 다르고 값이 같은 행을 접기 위한 열쇠. `source_id` 를 뺀다."""
        return (self.substance, self.species, self.threshold_type, self.dose, self.unit)


_NUM = re.compile(r"\d+(?:\.\d+)?")


def parse_low(dose: str) -> float | None:
    """문자열에서 **가장 낮은 수치**를 뽑는다. 없으면 `None`."""
    nums = [float(m.group()) for m in _NUM.finditer(dose or "")]
    return min(nums) if nums else None


def _table_path() -> Path:
    """설치 형태와 무관하게 표를 찾는다.

    `paths.find_root()` 를 쓰지 않는다 — 설치본에서는 루트가 없을 수 있고,
    이 표는 **패키지 데이터**(`pyproject.toml` 의 `compute/tables/*.csv`)이기 때문이다.
    """
    p = resources.files("pettriage.compute") / "tables" / TABLE_NAME
    with resources.as_file(p) as real:
        if real.exists():
            return real
    raise RuleTableMissingError(
        f"{TABLE_NAME} 를 찾지 못했다. `python scripts/build_rule_table.py --write` 로 생성할 것."
    )


@lru_cache(maxsize=1)
def load_rules() -> tuple[Rule, ...]:
    """표 전체. 프로세스당 한 번만 읽는다."""
    path = _table_path()
    out: list[Rule] = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.append(
                Rule(
                    fact_id=r["fact_id"],
                    substance=r["substance"],
                    species=r["species"],
                    threshold_type=r["threshold_type"],
                    dose=r["dose"],
                    unit=r["unit"],
                    computable=(r.get("computable") or "").strip().upper() == "Y",
                    effect=r.get("effect", ""),
                    signs=r.get("signs", ""),
                    onset=r.get("onset", ""),
                    source_id=r["source_id"],
                    citation=r.get("citation", ""),
                    note=r.get("note", ""),
                )
            )
    if not out:
        raise RuleTableMissingError(f"{path} 가 비었다. 사실 표를 확인하고 다시 생성할 것.")
    return tuple(out)


def _dedupe(rules: list[Rule]) -> list[Rule]:
    """값이 같고 출처만 다른 행을 하나로 접는다.

    양파가 `F-034-001`(S-034)·`F-098-002`(S-098) 에 **같은 `15-30 g/kg`** 로 있다.
    접지 않으면 같은 근거를 두 번 센다.
    """
    seen: dict[tuple[str, str, str, str, str], Rule] = {}
    for r in rules:
        seen.setdefault(r.key(), r)
    return list(seen.values())


def lookup(substance: str, species: str) -> list[Rule]:
    """(물질 × 종) 으로 조회. **없으면 빈 리스트다.**

    `substance` 는 부분 일치로 본다 — 표는 `초콜릿(테오브로민+카페인)` 로 적고
    질의는 `초콜릿` 으로 들어온다.
    """
    if not substance:
        return []
    want = SPECIES_WIDEN.get(species, (species,)) if species else None
    hit = [
        r
        for r in load_rules()
        if (substance in r.substance or r.substance in substance)
        and (want is None or r.species in want)
    ]
    return sorted(_dedupe(hit), key=lambda r: (r.severity, r.low if r.low is not None else 0.0))


def computable_for(substance: str, species: str) -> list[Rule]:
    """**체중과 곱해 판정할 수 있는 행만.**

    `computable=N` 행(백합 `1-2 leaves` · 소철 `2 seeds` · 주목 `2.3 g leaves/kg`)은
    여기서 빠진다. 원문이 개수로만 말했으므로 체중당 환산이 불가능하고,
    **잎 한 장의 무게를 우리가 정하면 그게 곧 환각이다.**
    """
    return [r for r in lookup(substance, species) if r.computable and r.low is not None]


def qualitative_for(substance: str, species: str) -> list[Rule]:
    """정량 판정은 못 하지만 **정성 문장으로는 말할 수 있는 행.**

    *"백합 잎 1-2장으로 중독이 보고되었다"* 는 말할 수 있고,
    *"체중 4kg 고양이가 X g 먹었으니 위험"* 은 말할 수 없다.
    """
    return [r for r in lookup(substance, species) if not r.computable]


def has_quantitative(substance: str, species: str) -> bool:
    """정량으로 답할 근거가 있나. **없으면 부르는 쪽이 정성 답변이나 거절로 내려간다** (D-46)."""
    return bool(computable_for(substance, species))


# ─────────────────────────────────────────────────────────────
# 바닥 등급 산출 (D-50)
# ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RuleVerdict:
    """규칙 테이블 1차 판정. **`level` 이 `None` 이면 정량 판정을 하지 않는다.**"""

    level: TriageLevel | None
    #: 넘긴 역치 행. 답변의 근거로 그대로 인용한다.
    crossed: tuple[Rule, ...] = ()
    #: `MONITOR` 일 때 함께 나가는 상승 조건. 비면 `apply_gate` 가 거부한다 (D-39).
    escalation_conditions: tuple[str, ...] = ()
    #: 출처 간 수치가 `CONFLICT_RATIO` 이상 벌어졌다.
    conflict: bool = False
    #: 사람이 읽는 사유. 로그·오류 분석에 쓴다 (04 §7).
    reason: str = ""


def to_mg_per_kg(dose: float, unit: str) -> float | None:
    """질량 단위를 `mg/kg` 으로. 환산할 수 없으면 `None`."""
    factor = _MG_PER_KG.get(unit.strip())
    return None if factor is None else dose * factor


def _signs_of(rules: list[Rule]) -> tuple[str, ...]:
    """여러 행의 증상을 합친다. **순서를 유지하고 중복만 접는다.**"""
    out: list[str] = []
    for r in rules:
        for sign in (x.strip() for x in r.signs.split("|")):
            if sign and sign not in out:
                out.append(sign)
    return tuple(out)


def _detect_conflict(rules: list[Rule]) -> str | None:
    """같은 역치종류끼리 최대/최소가 `CONFLICT_RATIO` 이상 벌어졌나.

    **역치종류를 넘어 비교하지 않는다** — `임상징후 발현 20` 과 `중증 60` 은
    상충이 아니라 **단계**다. 섞어 재면 정상 데이터를 상충으로 잡는다.
    """
    by_type: dict[str, list[float]] = {}
    for r in rules:
        if r.low is None:
            continue
        mg = to_mg_per_kg(r.low, r.unit)
        if mg is not None:
            by_type.setdefault(r.threshold_type, []).append(mg)
    for ttype, values in by_type.items():
        lo, hi = min(values), max(values)
        if lo > 0 and hi / lo >= CONFLICT_RATIO:
            return (
                f"'{ttype}' 값이 출처 간 {hi / lo:.0f}배 벌어졌다 "
                f"({lo:g}~{hi:g} mg/kg) — 정량 판정을 포기한다"
            )
    return None


def rule_level_for(substance: str, species: str, amount_mg_per_kg: float) -> RuleVerdict:
    """체중당 섭취량 → **바닥 등급** (D-50).

    Args:
        amount_mg_per_kg: 이미 `mg/kg` 으로 환산된 섭취량.
            `to_mg_per_kg()` 를 쓴다. 단위를 코드가 고르지 않는다 (D-17 후속).

    `level=None` 인 경우는 둘이다.

      1. 계산 가능한 역치가 없다 — 조류가 전부 여기다 (D-09)
      2. **출처 간 수치가 10배 이상 벌어졌다** — 어느 쪽이 맞는지 모르므로 포기한다

    둘 다 실패가 아니라 **정성 답변으로 내려가라는 신호**다 (D-46).
    """
    rules = computable_for(substance, species)
    if not rules:
        return RuleVerdict(None, reason=f"{substance}·{species} 에 계산 가능한 역치가 없다")

    conflict = _detect_conflict(rules)
    if conflict is not None:
        return RuleVerdict(None, conflict=True, reason=f"{substance}·{species} — {conflict}")

    normalized = {
        i: mg
        for i, r in enumerate(rules)
        if r.low is not None and (mg := to_mg_per_kg(r.low, r.unit)) is not None
    }
    crossed = [
        r for i, r in enumerate(rules) if i in normalized and amount_mg_per_kg >= normalized[i]
    ]
    if not crossed:
        # **역치 미만도 답한다.** 상승 조건을 붙여 MONITOR 로 내린다.
        conditions = _signs_of(rules)
        return RuleVerdict(
            BELOW_THRESHOLD_LEVEL if conditions else None,
            escalation_conditions=conditions,
            reason=(
                f"가장 낮은 역치 {min(normalized.values()):g} mg/kg 미만"
                if conditions
                else "역치 미만이나 상승 조건이 없어 등급을 매기지 않는다 (D-39)"
            ),
        )

    level = max(THRESHOLD_TO_LEVEL[r.threshold_type] for r in crossed)
    top = max(crossed, key=lambda r: SEVERITY[r.threshold_type])
    return RuleVerdict(
        level,
        crossed=tuple(crossed),
        reason=f"'{top.threshold_type}' 역치 {top.dose}{top.unit} 초과 ({top.fact_id})",
    )
