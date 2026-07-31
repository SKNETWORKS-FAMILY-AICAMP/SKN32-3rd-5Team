"""문장화 템플릿 — `doc_type`별로 데이터에서 도출.

설계 근거: docs/06_설계결정기록.md · D-38

    문장화를 LLM에게 시키지 않는다. 축① "결정론은 코드로"의 적용이다.
    3문을 돌리면 ①에서 끝난다 — 필드를 문장에 끼워 넣는 일에
    자연어 생성이 꼭 필요하지는 않다.

    템플릿은 발명하지 않고 코퍼스의 `doc_type` 축에서 도출했다.

핵심 규칙 두 가지:

  1. **결측 필드는 절을 통째로 뺀다.** "정보 없음"을 출력하지 않는다 —
     그 문장이 검색되면 그 자체가 오답이다.

  2. 그 결과 **임계값이 없으면 정량 문장이 생성되지 않는다.**
     조류는 정량 임계치가 0건이므로 조류 청크는 자동으로 정성 문장만 나온다.
     D-09의 종별 분기가 데이터 단계에서 강제된다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..schemas import Fact

# 한 절(clause) = (조건, 문장 생성기).
# 조건이 False면 그 절은 출력되지 않는다.
Clause = tuple[Callable[[Fact], bool], Callable[[Fact], str]]


@dataclass(frozen=True)
class Template:
    doc_type: str
    clauses: Sequence[Clause]

    def render(self, fact: Fact) -> str:
        parts = [make(fact) for cond, make in self.clauses if cond(fact)]
        return " ".join(p.strip() for p in parts if p and p.strip())


def _has(field: str) -> Callable[[Fact], bool]:
    return lambda f: bool(getattr(f, field, None))


def _cite(f: Fact) -> str:
    return f"(출처: {f.publisher}, {f.source_id})"


#: 역치로 말할 수 있는 임계치 종류. 나머지는 정량 문장을 만들지 않는다.
#:
#: `증례 보고 범위` 를 "N 이상 섭취 시" 로 쓰면 **출처에 없는 주장을 하게 된다.**
#: S-034 Table 1의 캡션은 "Range of doses ... reported to cause" 이고,
#: 실제로 용량-반응이 역전한다 (포도 3 g/kg 사망 vs 20.6 g/kg 회복).
THRESHOLD_TYPES = frozenset({"임상징후 발현", "중증", "치사"})


def _has_threshold(f: Fact) -> bool:
    return bool(f.dose) and bool(f.unit) and f.threshold_type in THRESHOLD_TYPES


def _has_reported_range(f: Fact) -> bool:
    return bool(f.dose) and bool(f.unit) and f.threshold_type == "증례 보고 범위"


def _dose_phrase(f: Fact) -> str:
    """단위가 이미 체중당(`mg/kg`)이면 "체중 1kg당"을 덧붙이지 않는다.

    붙이면 `체중 1kg당 20mg/kg` 이라는 이중 표기가 되어 수치가 왜곡된다.
    """
    unit = f.unit or ""
    if "/kg" in unit.replace(" ", ""):
        return f"{f.dose}{unit}"
    return f"체중 1kg당 {f.dose}{unit}"


TOXICITY_FOOD = Template(
    doc_type="toxicity_food",
    clauses=[
        # 급여 등급 절 — 등급이 없으면 생략한다.
        # "주의 대상"처럼 기본값을 채워 넣으면 출처에 없는 분류를 주장하게 된다.
        (
            _has("feeding_level"),
            lambda f: f"{f.species_ko}에게 {f.substance}은(는) {f.feeding_level_ko}로 분류된다.",
        ),
        (
            lambda f: not f.feeding_level,
            lambda f: f"{f.species_ko}와 {f.substance}에 관한 자료다.",
        ),
        # 정량 절 — 역치 성격이 확인된 값만. 조류는 임계치가 0건이라 항상 생략된다.
        (
            _has_threshold,
            lambda f: f"{_dose_phrase(f)} 이상 섭취 시 {f.effect_ko}이(가) 보고되었다.",
        ),
        # 증례 보고 범위는 역치가 아니다 — 범위로만 말한다
        (
            _has_reported_range,
            lambda f: (
                f"증례 보고에서 {_dose_phrase(f)} 범위의 섭취가 "
                f"{f.effect_ko}과(와) 함께 보고되었다."
            ),
        ),
        # 역치가 없다고 명시된 경우 — 포도처럼 용량-반응이 성립하지 않는 물질
        (
            lambda f: f.threshold_type == "역치 없음",
            lambda f: "안전한 최소 섭취량이 확립되어 있지 않아, 양과 무관하게 주의가 필요하다.",
        ),
        (_has("signs"), lambda f: f"주요 증상은 {', '.join(f.signs)}이다."),
        (_has("onset"), lambda f: f"증상은 {f.onset} 이내에 나타난다."),
        (lambda f: True, _cite),
    ],
)

TOXICITY_PLANT = Template(
    doc_type="toxicity_plant",
    clauses=[
        (
            lambda f: True,
            lambda f: (
                f"{f.substance}"
                + (f"({f.scientific_name})" if f.scientific_name else "")
                + f"은(는) {f.species_ko}에게 독성이 있다."
            ),
        ),
        (_has("toxic_part"), lambda f: f"{f.toxic_part}이(가) 독성 부위다."),
        (
            _has_threshold,
            lambda f: f"{f.dose}{f.unit} 섭취 시 {f.effect_ko}이(가) 보고되었다.",
        ),
        (
            _has_reported_range,
            lambda f: f"증례 보고에서 {f.dose}{f.unit} 범위가 {f.effect_ko}과(와) 함께 보고되었다.",
        ),
        (_has("signs"), lambda f: f"주요 증상은 {', '.join(f.signs)}이다."),
        (_has("onset"), lambda f: f"증상은 {f.onset} 이내에 나타난다."),
        (lambda f: True, _cite),
    ],
)

EMERGENCY = Template(
    doc_type="emergency",
    clauses=[
        (
            lambda f: True,
            lambda f: f"{f.species_ko}에서 {f.substance}은(는) {f.triage_ko} 상황이다.",
        ),
        (_has("signs"), lambda f: f"확인할 증상은 {', '.join(f.signs)}이다."),
        (
            _has("escalation_conditions"),
            lambda f: (
                "다음에 해당하면 즉시 병원에 연락한다 — " + ", ".join(f.escalation_conditions) + "."
            ),
        ),
        (lambda f: True, _cite),
    ],
)

SYMPTOM = Template(
    doc_type="symptom",
    clauses=[
        (
            lambda f: True,
            lambda f: f"{f.species_ko}의 {f.substance}은(는) 관찰이 필요한 징후다.",
        ),
        (_has("signs"), lambda f: f"함께 나타날 수 있는 증상은 {', '.join(f.signs)}이다."),
        (
            _has("escalation_conditions"),
            lambda f: "다음 경우 진료가 필요하다 — " + ", ".join(f.escalation_conditions) + ".",
        ),
        (lambda f: True, _cite),
    ],
)

NUTRITION = Template(
    doc_type="nutrition",
    clauses=[
        (
            lambda f: True,
            lambda f: f"{f.species_ko} {f.life_stage or '전 생애단계'}의 {f.substance} 권장량이다.",
        ),
        (
            lambda f: bool(f.dose) and bool(f.unit),
            lambda f: f"최소 {f.dose}{f.unit}가 권장된다.",
        ),
        (
            lambda f: bool(f.max_value) and bool(f.unit),
            lambda f: f"최대 허용량은 {f.max_value}{f.unit}다.",
        ),
        (_has("basis"), lambda f: f"기준은 {f.basis}다."),
        (lambda f: True, _cite),
    ],
)

RECALL = Template(
    doc_type="recall",
    clauses=[
        (lambda f: True, lambda f: f"{f.substance} 관련 리콜·안전 정보다."),
        (_has("signs"), lambda f: f"보고된 문제는 {', '.join(f.signs)}이다."),
        (_has("onset"), lambda f: f"발표 시점은 {f.onset}다."),
        (lambda f: True, _cite),
    ],
)

TEMPLATES: dict[str, Template] = {
    t.doc_type: t for t in (TOXICITY_FOOD, TOXICITY_PLANT, EMERGENCY, SYMPTOM, NUTRITION, RECALL)
}
