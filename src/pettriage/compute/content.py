"""성분 함량 — **먹은 것의 무게**를 **유효성분의 무게**로 바꾼다 (D-16 · D-78).

    초콜릿 20 g 을 먹었다  ≠  테오브로민 20 g 을 먹었다

    역치는 유효성분 기준이다 — `F-034-020` 은 *"개는 테오브로민·카페인 20 mg/kg 에서
    임상징후"* 라고 말한다. 그런데 계산 노드는 **초콜릿 무게**를 그 역치에 그대로 댔다.

        밀크초콜릿 20 g / 5 kg = 4 g/kg = 4,000 mg/kg  →  60 mg/kg(중증) 초과 → EMERGENCY
        실제 테오브로민       20 g × 2 mg/g = 40 mg
                             40 mg / 5 kg  =     8 mg/kg  →  20 mg/kg 미만 → MONITOR

    **200배 틀렸고, 방향이 과대였다.** 과대는 이 도메인에서 의도된 편향이지만(D-13),
    골든셋이 `MONITOR` 라고 적어 둔 질의에 *"지금 병원"* 을 내는 것은 편향이 아니라 오답이다.

## 이 표는 손으로 관리한다

    `정량임계치.csv`·`정성등급.csv` 는 `make rules` 가 사실 표에서 낸다. 이 표는 아니다 —
    **`active_substance` 가 사실 표에 없는 연결**이기 때문이다. `F-034-025`(밀크 초콜릿의
    함량)와 `F-034-020`(테오브로민 역치)이 같은 출처(S-034)의 같은 논문에서 왔다는 것은
    사람이 읽어야 아는 사실이다.

    그래서 `별칭.csv` 와 같은 지위다 — **손으로 적되 `basis` 를 남기고 검사기가 대조한다**
    (`tests/test_content.py` 가 사실 표의 수치와 한 줄씩 맞춰 본다). 근거 없는 계수는
    우리가 만든 숫자가 등급을 판정하게 두는 것이고, 그것이 D-16 이 막으려는 것이다.

## 수치가 없는 행을 0 으로 적지 않는다

    화이트 초콜릿의 원문은 *"테오브로민 급원으로서는 무의미한 수준으로 간주됨"* 이다.
    **숫자가 아니다.** 0 을 적으면 우리가 만든 숫자가 되고, `0 mg/kg → 안전` 이라는
    판정을 원문이 한 적 없는 강도로 말하게 된다.

    대신 `bound=무의미` 로 두고 **정량 판정을 건너뛴다.** 역치 행은 그대로 있으므로
    `apply_rule_table` 이 역치 미만(`MONITOR`) 으로 바닥을 깔고, 상승 조건이 함께 나간다.
    답은 같지만 **말할 수 있는 것만 말한 답**이다.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path

log = logging.getLogger(__name__)

TABLE_NAME = "성분함량.csv"

#: `mg_per_g` 를 어떻게 읽어야 하는가.
#:
#:   정확    원문이 값 하나를 준다 (`5 mg/g`)
#:   이상    원문이 부등호를 쓴다 (`>14 mg/g`) — **하한**이다. 실제는 더 높을 수 있다
#:   무의미  원문이 수치를 주지 않고 무시할 수준이라고만 말한다 → **정량 판정을 안 한다**
BOUNDS = ("정확", "이상", "무의미")


class ContentTableError(RuntimeError):
    """표가 없거나 규칙을 어겼다. **조용히 넘어가지 않는다** — 계수가 빠지면 200배 틀린다."""


@dataclass(frozen=True)
class Content:
    """(물질 × 종) 한 행."""

    substance: str
    species: tuple[str, ...]
    #: 역치 표에서 **실제로 찾아볼 이름.** 함량 행의 물질에는 역치가 없다 —
    #: `밀크 초콜릿` 에 mg/kg 역치는 없고, 그 역치는 `초콜릿(테오브로민+카페인)` 에 있다.
    active_substance: str
    mg_per_g: float | None
    bound: str
    basis: tuple[str, ...]
    source_id: str
    note: str = ""

    @property
    def quantifiable(self) -> bool:
        """정량 판정을 할 수 있는가. `무의미` 는 **못 한다** — 0 이 아니라 모른다."""
        return self.mg_per_g is not None and self.bound != "무의미"

    @property
    def negligible(self) -> bool:
        """**자료가 "무의미한 수준"이라고 말했다** — 모르는 것이 아니라 확인된 사실이다.

        🔴 이 구분이 판정을 가른다. 2026-08-02 실측 (G-041) —

            "강아지가 화이트초콜릿 100g을 먹었는데"   ← 체중이 문장에 없다
            → 양 미상 바닥(D-79)이 걸려 CALL_NOW
            → LLM 은 자료를 읽고 MONITOR 라 했고, 게이트가 그것을 막았다
            → 골든셋 기대: MONITOR

        바닥의 논리는 *"독성이 확인된 물질 + 양 미상 → 전화해서 확인"* 인데,
        **유효성분 함량 자체가 무의미하면 그 전제가 성립하지 않는다.** 체중을 알든
        모르든 유효성분 용량은 무의미하다 — **물어볼 이유가 없는 것을 모른다고
        등급을 올린 것**이다.

        `quantifiable` 과 뜻이 겹쳐 보이지만 묻는 것이 다르다 —
        `quantifiable` 은 *"계산할 수 있나"*, 이것은 *"왜 못 하나"* 다.
        """
        return self.bound == "무의미"

    @property
    def is_lower_bound(self) -> bool:
        """실제 값이 이보다 **높을 수 있다.** 낮게 잡히면 그것이 곧 과소평가다 (D-13)."""
        return self.bound == "이상"

    def applies_to(self, species: str | None) -> bool:
        return not self.species or not species or species in self.species

    def active_mg(self, amount_g: float) -> float | None:
        """먹은 무게(g) → 유효성분(mg). 정량 불가면 `None`."""
        if not self.quantifiable or self.mg_per_g is None:
            return None
        return amount_g * self.mg_per_g


def _table_path() -> Path:
    p = resources.files("pettriage.compute") / "tables" / TABLE_NAME
    with resources.as_file(p) as real:
        if real.exists():
            return real
    raise ContentTableError(f"{TABLE_NAME} 를 찾지 못했다 (compute/tables/).")


@lru_cache(maxsize=1)
def load_contents() -> tuple[Content, ...]:
    """표 전체. 규칙을 어긴 행은 **전체를 세운다** (`별칭.csv` 와 같은 처리)."""
    out: list[Content] = []
    with _table_path().open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            substance = (r.get("substance") or "").strip()
            if not substance:
                continue
            bound = (r.get("bound") or "").strip()
            if bound not in BOUNDS:
                raise ContentTableError(
                    f"{TABLE_NAME}: {substance!r} 의 bound 가 {BOUNDS} 가 아니다: {bound!r}"
                )
            raw = (r.get("mg_per_g") or "").strip()
            mg: float | None = None
            if raw:
                try:
                    mg = float(raw)
                except ValueError as e:
                    raise ContentTableError(
                        f"{TABLE_NAME}: {substance!r} 의 mg_per_g 를 못 읽는다: {raw!r}. "
                        "부등호는 bound 열에 적고 여기에는 숫자만 둔다."
                    ) from e
            if bound != "무의미" and mg is None:
                raise ContentTableError(
                    f"{TABLE_NAME}: {substance!r} 의 bound 가 {bound!r} 인데 수치가 없다. "
                    "수치를 못 적겠으면 bound 를 '무의미' 로 둔다 — 0 을 지어내지 않는다."
                )
            active = (r.get("active_substance") or "").strip()
            if not active:
                raise ContentTableError(
                    f"{TABLE_NAME}: {substance!r} 에 active_substance 가 없다. "
                    "**함량만 알고 역치를 모르면 계산할 수 없다.**"
                )
            basis = tuple(b for b in (r.get("basis") or "").split("|") if b)
            if not basis:
                raise ContentTableError(
                    f"{TABLE_NAME}: {substance!r} 에 basis 가 없다. "
                    "근거 없는 계수는 **우리가 만든 숫자가 등급을 판정하게** 둔다 (D-16)."
                )
            out.append(
                Content(
                    substance=substance,
                    species=tuple(
                        s for s in (r.get("species") or "").replace(",", "|").split("|") if s
                    ),
                    active_substance=active,
                    mg_per_g=mg,
                    bound=bound,
                    basis=basis,
                    source_id=(r.get("source_id") or "").strip(),
                    note=(r.get("note") or "").strip(),
                )
            )
    if not out:
        raise ContentTableError(f"{_table_path()} 가 비었다.")
    return tuple(out)


def content_for(substance: str | None, species: str | None) -> Content | None:
    """(물질 × 종) 에 함량 계수가 있나. **없으면 `None`** — 그때는 예전대로 돈다."""
    if not substance:
        return None
    for c in load_contents():
        if c.substance == substance and c.applies_to(species):
            return c
    return None


def threshold_substance(substance: str | None, species: str | None) -> str | None:
    """역치 표에서 찾아볼 이름. 함량 행이 없으면 **받은 이름 그대로.**

    `밀크 초콜릿` 은 어휘표에 있지만 역치 행이 0개다. 이 함수를 안 거치면
    `apply_rule_table` 이 빈 손으로 돌아오고 **판정불가 거절**이 된다 —
    물질을 정확히 알아낸 질의가 그래서 거절되면 앞 단계가 다 헛일이다.
    """
    c = content_for(substance, species)
    return c.active_substance if c else substance
