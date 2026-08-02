"""사실 표 · 청크 스키마.

설계 근거: docs/06_설계결정기록.md · D-14 · D-15 · D-17 · D-37 · D-38 · D-39

경로 ②(사실 추출 + 자체 문장화)가 기본 경로이므로, 벡터DB에 들어가는 것은
원문이 아니라 이 `Fact` 를 템플릿에 통과시킨 **우리 문장**이다.
원문은 data_work/ 에 남아 골든셋 채점 기준으로만 쓰인다 (D-29).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .triage.levels import FeedingLevel, TriageLevel

Species = Literal["dog", "cat", "bird", "mammal", "all"]
DocType = Literal["toxicity_food", "toxicity_plant", "nutrition", "emergency", "symptom", "recall"]

#: 임계치의 **성격**. 이 필드가 없으면 증례 보고 범위를 역치로 오인한다.
#:
#: S-034 실측에서 드러난 문제 — Table 1의 g/kg 값들은 캡션이
#: "Range of doses ... reported to cause"로, **보고된 증례 용량**이지 역치가 아니다.
#: 실제로 용량-반응이 역전한다 (포도 3 g/kg 사망 vs 20.6 g/kg 회복).
ThresholdType = Literal[
    "임상징후 발현",
    "중증",
    "치사",
    "증례 보고 범위",  # ⚠️ 역치 아님. 규칙 테이블에 넣지 말 것
    "성분 함량",
    "역치 없음",  # 용량-반응 미성립이 원문에 명시된 경우
    "기타",
]

SPECIES_KO: dict[str, str] = {
    "dog": "개",
    "cat": "고양이",
    "bird": "앵무새",
    "mammal": "개·고양이",
    "all": "반려동물",
}


def _unpipe(value: str | None) -> str:
    """`|` 로 이어 붙인 값을 읽을 수 있는 문장 조각으로 바꾼다.

    `"잎|꽃잎|꽃가루"` → `"잎, 꽃잎, 꽃가루"`. 값이 없으면 빈 문자열이다.
    """
    if not value:
        return ""
    return ", ".join(p.strip() for p in value.split("|") if p.strip())


@dataclass
class Fact:
    """원문에서 추출한 사실 1건. 문장이 아니라 **필드**다.

    검증 대상이 문장이 아니라 필드이므로 자동 대조가 가능하다 (D-38 · 04 §2.5).
    """

    # ── 식별 ────────────────────────────────────────────────
    fact_id: str
    source_id: str
    publisher: str
    doc_type: DocType
    species: Species

    # ── 대상 ────────────────────────────────────────────────
    substance: str
    scientific_name: str | None = None  # 학명 — 언어 무관 검증 앵커
    toxic_part: str | None = None

    # ── 정량 ────────────────────────────────────────────────
    threshold_type: ThresholdType | None = None
    dose: str | None = None
    unit: str | None = None
    max_value: str | None = None
    basis: str | None = None  # 예: "1,000 kcal ME 당"
    life_stage: str | None = None

    # ── 판정 ────────────────────────────────────────────────
    feeding_level: FeedingLevel | None = None  # 축 B
    triage_level: TriageLevel | None = None  # 축 A
    effect: str | None = None
    signs: list[str] = field(default_factory=list)
    onset: str | None = None
    escalation_conditions: list[str] = field(default_factory=list)

    # ── 역추적 (04 §2.5 역추적 가능성 = 100% 목표) ───────────
    quote: str | None = None  # 원문 문장. 경로 ①만 채운다
    locator: str | None = None  # 페이지·절·행
    accessed_at: str | None = None
    citation: str | None = None  # 원문이 인용한 1차 출처 번호

    # ── 파생 표현 (템플릿이 쓴다) ─────────────────────────────
    @property
    def species_ko(self) -> str:
        return SPECIES_KO.get(self.species, "반려동물")

    @property
    def feeding_level_ko(self) -> str:
        """급여 등급 표기. **등급이 없으면 빈 문자열** — 템플릿이 절을 생략한다.

        "주의 대상" 같은 기본값을 채우면 출처에 없는 분류를 주장하게 된다.
        """
        return self.feeding_level.label if self.feeding_level else ""

    @property
    def triage_ko(self) -> str:
        """등급 배지. **없으면 빈 문자열이다 — 기본값을 만들지 않는다.**

        예전에는 `"확인 필요"` 를 돌려주었다. 그런데 그것은 4등급 어디에도 없는 말이고,
        **출처가 등급을 주지 않은 자료에 우리가 등급을 붙이는 것**이었다.

            "앵무새에서 향초·왁스멜트·플러그인 방향제(VOC)는 **확인 필요 상황이다**."
            ← S-078 은 등급을 표기하지 않는다. emergency 131행 중 **82행**이 이랬다.

        `TOXICITY_FOOD` 의 급여 등급 절에는 같은 규율이 이미 적혀 있었다 —
        *"등급이 없으면 생략한다. 기본값을 채워 넣으면 출처에 없는 분류를 주장하게 된다."*
        `EMERGENCY` 에만 적용되지 않았다 (2026-08-01 골든셋 검수에서 발견).
        """
        return self.triage_level.badge if self.triage_level else ""

    @property
    def effect_ko(self) -> str:
        return _unpipe(self.effect) or "임상 징후"

    @property
    def toxic_part_ko(self) -> str:
        """독성 부위. **`|` 를 문장 부호로 바꾼다.**

        `|` 는 CSV 안에서 여러 값을 담기 위한 **직렬화 구분자**이지 문장 부호가 아니다.
        `signs`·`escalation_conditions` 는 `facts_io.LIST_FIELDS` 가 이미 쪼개는데
        `toxic_part`·`effect` 는 목록에 없어서 그대로 새어 나갔다 —
        *"잎|꽃잎|꽃가루|꽃병 물이 독성 부위다"* 가 18청크에서 실측됐다 (2026-08-02).
        임베딩과 화면 양쪽을 해친다.
        """
        return _unpipe(self.toxic_part)


@dataclass
class Chunk:
    """벡터DB에 적재되는 단위. 물질(항목) 단위 청킹 (D-14)."""

    chunk_id: str
    text: str  # 경로 ②는 템플릿 생성 문장, 경로 ①은 원문
    source_id: str
    species: Species
    doc_type: DocType
    #: 발행처. **인덱스가 들고 있어야 한다** — `Citation.publisher` 가 필수인데
    #: 청크에 없으면 검색 결과로 근거를 만들 수 없다. `source_id` 만으로는
    #: 되찾을 방법이 저장소 어디에도 없었다 (2026-08-02 검토).
    #: 기존 인덱스와의 호환을 위해 기본값을 둔다 — 재적재하면 채워진다.
    publisher: str = ""
    substance: str | None = None
    route: Literal["원문적재", "사실추출"] = "사실추출"
    quote: str | None = None  # 경로 ②는 비운다 (D-37)
    locator: str | None = None
    fact_ids: list[str] = field(default_factory=list)
