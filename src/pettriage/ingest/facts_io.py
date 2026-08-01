"""사실 표 CSV → `Fact` → 청크.

설계 근거: docs/01e_사실표작성지침.md · docs/06 D-14 · D-37 · D-38

    사람이 채우는 것은 CSV 한 장뿐이고, 그 아래는 전부 코드가 한다.

        facts_*.csv ──[load_facts]──→ Fact ──[to_chunk]──→ Chunk ──→ 벡터DB

    문장화에 LLM을 부르지 않는다 (D-38). 표가 맞으면 문장도 맞는다.

파일이 4개(팀원별)로 나뉘어 있으므로 병합도 여기서 한다.
`fact_id` 중복은 **조용히 덮지 않고 예외로 터뜨린다** — 덮으면 어느 쪽이
살아남았는지 알 수 없고, 그 상태로 적재되면 추적이 불가능해진다.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..schemas import Chunk, Fact
from ..triage.levels import FeedingLevel, TriageLevel
from .verbalize import to_chunk

#: `|` 로 여러 값을 담는 칸 (01e §4)
LIST_FIELDS = ("signs", "escalation_conditions")

#: CSV 에는 있지만 `Fact` 에는 없는 칸 — 작업 관리용이라 버린다
META_FIELDS = ("extracted_by", "note")


class DuplicateFactId(ValueError):
    """`fact_id` 가 겹친다. 병합 전에 고쳐야 한다."""


def _clean(v: str | None) -> str | None:
    v = (v or "").strip()
    return v or None


def row_to_fact(row: dict[str, str]) -> Fact:
    """CSV 한 행 → `Fact`. 빈 칸은 `None` 으로 남는다.

    빈 칸을 기본값으로 채우지 않는 것이 핵심이다 — 템플릿이 그 절을 생략하고,
    그래서 "정보 없음" 같은 문장이 벡터DB에 들어가지 않는다 (D-38).
    """
    data: dict[str, object] = {}
    for key, raw in row.items():
        key = (key or "").strip()
        if not key or key in META_FIELDS:
            continue
        if key in LIST_FIELDS:
            data[key] = [p.strip() for p in (raw or "").split("|") if p.strip()]
            continue
        val = _clean(raw)
        if val is None:
            continue
        if key == "feeding_level":
            data[key] = FeedingLevel[val]
        elif key == "triage_level":
            data[key] = TriageLevel[val]
        else:
            data[key] = val

    for f in LIST_FIELDS:
        data.setdefault(f, [])
    return Fact(**data)  # type: ignore[arg-type]


def load_facts(path: Path) -> list[Fact]:
    """CSV 한 파일을 읽는다. 엑셀이 붙이는 BOM 을 감안한다."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if any((v or "").strip() for v in r.values())]
    return [row_to_fact(r) for r in rows]


def load_all(facts_dir: Path, pattern: str = "facts_*.csv") -> list[Fact]:
    """팀원별 파일을 모두 읽어 병합한다.

    Raises:
        DuplicateFactId: `fact_id` 가 겹칠 때. 어느 쪽을 살릴지는 사람이 정한다.
    """
    facts: list[Fact] = []
    for p in sorted(facts_dir.glob(pattern)):
        facts.extend(load_facts(p))

    dup = [k for k, n in Counter(f.fact_id for f in facts).items() if n > 1]
    if dup:
        raise DuplicateFactId(
            f"fact_id 중복 {len(dup)}건: {sorted(dup)[:5]} — "
            "조용히 덮으면 어느 쪽이 적재됐는지 추적할 수 없다"
        )
    return facts


def build_chunks(facts: Iterable[Fact]) -> list[Chunk]:
    """사실 → 템플릿 문장 → 청크. 물질(항목) 단위다 (D-14)."""
    return [to_chunk(f) for f in facts]


def summarize(facts: Sequence[Fact]) -> dict[str, dict[str, int]]:
    """적재 전 분포. **한쪽으로 쏠렸는지 눈으로 확인하라고** 내보낸다.

    04 §2.3 이 종별 최소 건수를 요구하므로, 여기서 조류가 0이면
    골든셋 설계 자체가 성립하지 않는다.
    """
    return {
        "species": dict(Counter(f.species for f in facts)),
        "doc_type": dict(Counter(f.doc_type for f in facts)),
        "threshold_type": dict(Counter(f.threshold_type or "(없음)" for f in facts)),
        "source": dict(Counter(f.source_id for f in facts)),
    }
