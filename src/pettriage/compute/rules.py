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

TABLE_NAME = "정량임계치.csv"

#: 종 질의를 넓히는 규칙. `mammal` 은 개·고양이 공통 값, `all` 은 종 무관이다.
SPECIES_WIDEN: dict[str, tuple[str, ...]] = {
    "dog": ("dog", "mammal", "all"),
    "cat": ("cat", "mammal", "all"),
    "bird": ("bird", "all"),
}

#: 심각도 순서. 같은 물질에 여러 역치가 있으면 낮은 것부터 넘는다.
SEVERITY: dict[str, int] = {"임상징후 발현": 1, "중증": 2, "치사": 3}


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
