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


def _strip_trailing_paren(word: str) -> str:
    """끝에 붙은 괄호 묶음을 떼어낸다. 중첩·다중 괄호도 반복해서 벗긴다."""
    w = word.rstrip()
    while w.endswith(")") or w.endswith("）"):
        depth, cut = 0, None
        for i in range(len(w) - 1, -1, -1):
            if w[i] in ")）":
                depth += 1
            elif w[i] in "(（":
                depth -= 1
                if depth == 0:
                    cut = i
                    break
        if cut is None or cut == 0:  # 여는 괄호가 없거나 통째로 괄호면 그대로 둔다
            return w
        w = w[:cut].rstrip()
    return w or word


def _has_batchim(word: str) -> bool | None:
    """마지막 글자에 받침이 있는가. 판단할 수 없으면 `None`.

    한글 음절은 유니코드에서 `0xAC00 + (초성*588 + 중성*28 + 종성)` 으로 배치된다.
    따라서 `(코드 - 0xAC00) % 28` 이 0이 아니면 받침이 있다.

    **끝의 괄호 묶음은 통째로 무시한다.** 조사는 괄호가 아니라 그 앞의 말로 고른다 —
    `진통제(…아세트아미노펜)은` 이 아니라 `진통제(…아세트아미노펜)는` 이다.
    """
    word = _strip_trailing_paren(word)
    for ch in reversed(word.strip()):
        if ch.isspace() or ch in "[]{}<>·,.'\"":
            continue
        if "가" <= ch <= "힣":
            return (ord(ch) - 0xAC00) % 28 != 0
        if ch.isdigit():
            # 숫자는 읽는 소리로 판단한다 — 0·1·3·6·7·8 이 받침으로 끝난다.
            return ch in "013678"
        return None  # 영문·기호로 끝나면 판단하지 않는다
    return None


def _josa(word: str, with_batchim: str, without: str) -> str:
    """조사를 골라 붙인다. 판단이 안 되면 `은(는)` 형태로 병기한다.

    벡터DB에 들어가는 문장이자 **사용자가 읽는 문장**이다.
    `아보카도은(는)` 같은 표기는 검색 임베딩과 가독성을 함께 해친다.
    """
    b = _has_batchim(word)
    if b is None:
        return f"{word}{with_batchim}({without})"
    return f"{word}{with_batchim if b else without}"


def _eun(w: str) -> str:
    return _josa(w, "은", "는")


def _ga(w: str) -> str:
    return _josa(w, "이", "가")


def _ida(w: str) -> str:
    """서술격 조사. 받침이 있으면 `이다`, 없으면 `다`.

    `기준은 1,000kcal 대사에너지 당다` 처럼 붙여 쓰면 문장이 깨진다.
    """
    return _josa(w, "이다", "다")


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


def _is_composition(f: Fact) -> bool:
    """**성분 조성**이지 권장량이 아니다.

    같은 `doc_type=nutrition` 안에 성격이 다른 두 가지가 섞여 있다 —
    S-043의 영양소 **권장량**과 S-044의 원료 **성분 함량**이다.

    구분하지 않으면 *"어류기름 최소 100.71%가 권장된다"* 같은 문장이 나온다.
    원문에 없는 주장이고, 그대로 벡터DB에 들어가면 그 자체가 환각의 출처다 (D-38).
    """
    return f.threshold_type == "성분 함량"


TOXICITY_FOOD = Template(
    doc_type="toxicity_food",
    clauses=[
        # 급여 등급 절 — 등급이 없으면 생략한다.
        # "주의 대상"처럼 기본값을 채워 넣으면 출처에 없는 분류를 주장하게 된다.
        (
            _has("feeding_level"),
            lambda f: f"{f.species_ko}에게 {_eun(f.substance)} {f.feeding_level_ko}로 분류된다.",
        ),
        (
            lambda f: not f.feeding_level,
            lambda f: f"{f.species_ko}와 {f.substance}에 관한 자료다.",
        ),
        # 정량 절 — 역치 성격이 확인된 값만. 조류는 임계치가 0건이라 항상 생략된다.
        (
            _has_threshold,
            lambda f: f"{_dose_phrase(f)} 이상 섭취 시 {_ga(f.effect_ko)} 보고되었다.",
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
        # 성분 함량 — **섭취 역치가 아니라 그 식품에 독성 성분이 얼마나 들었는가.**
        #
        # 이 절이 없으면 초콜릿 종류별 테오브로민 함량(다크 5mg/g vs 밀크 2mg/g)이
        # 문장에서 통째로 사라진다. "다크가 왜 더 위험한가"의 근거가 그 수치다.
        (
            lambda f: _is_composition(f) and bool(f.dose) and bool(f.unit),
            lambda f: f"{f.dose}{f.unit} 수준의 함량이 보고되었다.",
        ),
        (
            lambda f: _is_composition(f) and not f.dose and bool(f.effect),
            lambda f: f"{f.effect}.",
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
            # 조사는 **학명 괄호가 아니라 물질명**으로 고른다 —
            # "백합(Lilium)는" 이 아니라 "백합(Lilium)은" 이 맞다.
            lambda f: (
                f"{f.substance}"
                + (f"({f.scientific_name})" if f.scientific_name else "")
                + f"{'은' if _has_batchim(f.substance) else '는'} {f.species_ko}에게 독성이 있다."
            ),
        ),
        (_has("toxic_part"), lambda f: f"{_ga(f.toxic_part)} 독성 부위다."),
        (
            _has_threshold,
            lambda f: f"{f.dose}{f.unit} 섭취 시 {_ga(f.effect_ko)} 보고되었다.",
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
            lambda f: f"{f.species_ko}에서 {_eun(f.substance)} {f.triage_ko} 상황이다.",
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
            lambda f: f"{f.species_ko}의 {_eun(f.substance)} 관찰이 필요한 징후다.",
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
        # 권장량 — 기준표에서 온 것
        (
            lambda f: not _is_composition(f),
            lambda f: f"{f.species_ko} {f.life_stage or '전 생애단계'}의 {f.substance} 권장량이다.",
        ),
        (
            lambda f: bool(f.dose) and bool(f.unit) and not _is_composition(f),
            lambda f: f"최소 {f.dose}{f.unit}가 권장된다.",
        ),
        (
            lambda f: bool(f.max_value) and bool(f.unit) and not _is_composition(f),
            lambda f: f"최대 허용량은 {f.max_value}{f.unit}다.",
        ),
        # 성분 조성 — 급여 기준이 아니라 그 물질에 무엇이 얼마나 들었는가
        (
            _is_composition,
            lambda f: f"{f.substance}의 성분 함량 정보다.",
        ),
        (
            lambda f: _is_composition(f) and bool(f.dose) and bool(f.unit),
            lambda f: (
                (f"{f.basis} " if f.basis else "")
                + f"{f.dose}{f.unit} 수준으로 보고되었다."
                # `effect_ko` 는 비었을 때 "임상 징후"를 돌려준다 — 성분 조성에는 없는 말이다.
                # 원본 필드로 판단한다.
                + (f" {f.effect}." if f.effect else "")
            ),
        ),
        (
            lambda f: bool(f.basis) and not _is_composition(f),
            lambda f: f"기준은 {_ida(f.basis)}.",
        ),
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
