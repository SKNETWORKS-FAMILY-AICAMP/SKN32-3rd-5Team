#!/usr/bin/env python
"""사실 표 검사 — 커밋 전에 돌린다.

    python scripts/check_facts.py data/facts/facts_ohb.csv
    python scripts/check_facts.py                    # data/facts/*.csv 전부

설계 근거: docs/01e_사실표작성지침.md · docs/06 D-09 · D-37 · D-38 · D-39

여기서 잡는 것은 **사람이 손으로 쓰다가 내는 오류**다.
이 표에서 틀리면 벡터DB 문장과 트리아지 규칙 테이블이 같이 틀어지므로,
적재 전에 한 번 거른다 (04 §2.5 층 0).
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FACTS_DIR = ROOT / "data" / "facts"

REQUIRED = ("fact_id", "source_id", "publisher", "doc_type", "species", "substance", "locator")

DOC_TYPES = {"toxicity_food", "toxicity_plant", "nutrition", "emergency", "symptom", "recall"}
SPECIES = {"dog", "cat", "bird", "mammal", "all"}
FEEDING = {"NEVER", "CAUTION", "SAFE"}
TRIAGE = {"EMERGENCY", "CALL_NOW", "VISIT_SOON", "MONITOR"}
THRESHOLD_TYPES = {
    "임상징후 발현",
    "중증",
    "치사",
    "증례 보고 범위",
    "성분 함량",
    "역치 없음",
    "기타",
}
#: 규칙 테이블에 넣어도 되는 임계치 종류. 나머지는 정량 문장을 만들지 않는다.
USABLE_THRESHOLDS = {"임상징후 발현", "중증", "치사"}

#: 원문 적재가 허용된 자료 (D-37 판정). 이 목록 밖의 자료에 quote 를 실으면 안 된다.
ROUTE1_SOURCES = {"S-001", "S-012", "S-023", "S-042", "S-043", "S-055", "S-064", "S-070"}


@dataclass
class Issue:
    level: str  # ERROR | WARN
    where: str
    message: str

    def __str__(self) -> str:
        icon = "✗" if self.level == "ERROR" else "⚠"
        return f"  {icon} [{self.where}] {self.message}"


def _split(value: str) -> list[str]:
    return [p.strip() for p in (value or "").split("|") if p.strip()]


def check_row(row: dict[str, str], where: str) -> list[Issue]:
    out: list[Issue] = []
    g = lambda k: (row.get(k) or "").strip()  # noqa: E731

    # ── 필수 ────────────────────────────────────────────────
    for f in REQUIRED:
        if not g(f):
            out.append(Issue("ERROR", where, f"필수 칸이 비었다: {f}"))

    # ── 허용값 ──────────────────────────────────────────────
    if g("doc_type") and g("doc_type") not in DOC_TYPES:
        out.append(Issue("ERROR", where, f"doc_type 오타: {g('doc_type')!r} — {sorted(DOC_TYPES)}"))
    if g("species") and g("species") not in SPECIES:
        out.append(Issue("ERROR", where, f"species 오타: {g('species')!r} — {sorted(SPECIES)}"))
    if g("feeding_level") and g("feeding_level") not in FEEDING:
        out.append(Issue("ERROR", where, f"feeding_level 오타: {g('feeding_level')!r}"))
    if g("triage_level") and g("triage_level") not in TRIAGE:
        out.append(Issue("ERROR", where, f"triage_level 오타: {g('triage_level')!r}"))
    if g("threshold_type") and g("threshold_type") not in THRESHOLD_TYPES:
        out.append(Issue("ERROR", where, f"threshold_type 오타: {g('threshold_type')!r}"))

    # ── 정량 ────────────────────────────────────────────────
    dose, unit, ttype = g("dose"), g("unit"), g("threshold_type")

    if dose and not ttype:
        # 성격을 모르는 수치는 역치로 오인된다. 지침 3장.
        out.append(
            Issue("ERROR", where, "수치가 있는데 threshold_type 이 비었다 — 지침 3장을 볼 것")
        )
    if dose and not unit:
        out.append(
            Issue("ERROR", where, "dose 가 있는데 unit 이 비었다 (mg/kg 과 g/kg 은 1,000배)")
        )
    if unit and not dose:
        out.append(Issue("WARN", where, "unit 만 있고 dose 가 없다"))

    if dose and ttype == "증례 보고 범위":
        out.append(
            Issue(
                "WARN",
                where,
                "증례 보고 범위 — 역치가 아니다. 규칙 테이블에서 제외되고 "
                "'증례 보고에서 …' 문장으로 나간다. 의도한 것이면 무시할 것",
            )
        )

    # 조류는 체중당 임계치 자료가 0건이다 (D-09 개정)
    if g("species") == "bird" and dose:
        out.append(
            Issue(
                "ERROR",
                where,
                "조류에 dose 가 채워졌다 — 코퍼스에 조류 체중당 임계치는 0건이다. "
                "원문에 정말 있으면 note 에 출처 위치를 적고 팀장에게 알릴 것",
            )
        )

    # ── 판정 ────────────────────────────────────────────────
    if g("triage_level") == "MONITOR" and not _split(g("escalation_conditions")):
        out.append(
            Issue(
                "ERROR",
                where,
                "MONITOR 인데 escalation_conditions 가 비었다 — "
                "조건 없는 '관찰'은 과소평가로 채점된다 (D-39)",
            )
        )
    if g("species") == "bird" and g("feeding_level") == "SAFE":
        out.append(
            Issue(
                "ERROR",
                where,
                "조류에 SAFE 를 쓰지 않는다 — 출처끼리 티어가 충돌한다 (D-39). "
                "NEVER 또는 CAUTION 만",
            )
        )

    # ── 경로 ② 는 원문을 싣지 않는다 (D-37) ──────────────────
    if g("quote") and g("source_id") not in ROUTE1_SOURCES:
        out.append(
            Issue(
                "ERROR",
                where,
                f"{g('source_id')} 는 사실추출 한정 자료다 — quote 를 비울 것 (D-37)",
            )
        )

    # ── 단위 오식 ───────────────────────────────────────────
    if unit and unit.replace(" ", "") not in {
        "mg/kg",
        "g/kg",
        "mg",
        "g",
        "kg",
        "mL/kg",
        "mL",
        "%",
        "ppm",
        "IU/kg",
        "kcal/kg",
    }:
        out.append(Issue("WARN", where, f"보기 드문 단위: {unit!r} — 원문과 대조할 것"))

    return out


def check_file(path: Path) -> tuple[list[Issue], list[dict[str, str]]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    issues: list[Issue] = []
    for i, row in enumerate(rows, start=2):  # 헤더가 1행
        issues += check_row(row, f"{path.name}:{i}")
    return issues, rows


def check_cross(all_rows: list[dict[str, str]]) -> list[Issue]:
    """파일 간 검사 — 병합 시점에 터지는 것들을 미리 잡는다."""
    out: list[Issue] = []

    dup = [k for k, v in Counter(r.get("fact_id", "") for r in all_rows).items() if v > 1 and k]
    for k in sorted(dup):
        out.append(Issue("ERROR", "병합", f"fact_id 중복: {k}"))

    # 같은 (물질 · 종) 에 서로 다른 급여 등급이 붙으면 어느 쪽이 맞는지 정해야 한다
    grades: dict[tuple[str, str], set[str]] = defaultdict(set)
    for r in all_rows:
        lv = (r.get("feeding_level") or "").strip()
        if lv:
            grades[(r.get("substance", ""), r.get("species", ""))].add(lv)
    for (sub, sp), lv in sorted(grades.items()):
        if len(lv) > 1:
            out.append(
                Issue("WARN", "병합", f"{sub}({sp}) 급여 등급이 엇갈린다: {sorted(lv)} — 검수 필요")
            )
    return out


def main() -> int:
    args = sys.argv[1:]
    paths = [Path(a) for a in args] if args else sorted(FACTS_DIR.glob("facts_*.csv"))
    if not paths:
        print(f"검사할 파일이 없다. {FACTS_DIR} 에 facts_*.csv 를 만들 것.")
        return 1

    print("사실 표 검사 (01e 지침)\n")
    issues: list[Issue] = []
    all_rows: list[dict[str, str]] = []

    for p in paths:
        if not p.exists():
            print(f"  ✗ 파일 없음: {p}")
            return 1
        file_issues, rows = check_file(p)
        all_rows += rows
        issues += file_issues
        print(f"[{p.name}]  {len(rows)}행")
        for it in file_issues:
            print(it)
        if not file_issues:
            print("  · 문제 없음")
        print()

    if len(paths) > 1:
        cross = check_cross(all_rows)
        issues += cross
        print("[파일 간]")
        for it in cross or [Issue("WARN", "병합", "문제 없음")]:
            print(it if cross else "  · 문제 없음")
        print()

    errors = sum(1 for i in issues if i.level == "ERROR")
    warns = sum(1 for i in issues if i.level == "WARN")
    counts = Counter(r.get("species", "?") for r in all_rows)
    print(f"→ 총 {len(all_rows)}행 · 종별 {dict(counts)}")
    print(f"→ ERROR {errors} · WARN {warns}")
    if errors:
        print("\n  ERROR 를 고치고 커밋한다. 지침: docs/01e_사실표작성지침.md")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
