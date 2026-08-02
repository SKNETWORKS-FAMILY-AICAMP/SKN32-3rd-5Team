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
MANIFEST_DIR = ROOT / "data" / "manifests"

#: 인용 근거로 쓸 수 있는 스냅샷 품질 (D-30). `요약`·`빈약`·`부적합` 은 제외된다.
CITABLE_QUALITY = {"완전", "부분"}

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

#: 코퍼스에 실제로 나오는 단위. **환산하지 않고 원문 표기를 그대로 쓴다** (01e 규칙 2·3).
#:
#: 목록에 없으면 WARN 이다 — 막으려는 게 아니라 **오식을 눈에 띄게** 하려는 것이다.
#: `mg/kg` 과 `g/kg` 은 1,000배이고 실제로 S-034 에서 그 오류가 나왔다.
KNOWN_UNITS = {
    # 중독 — 체중당·절대량
    "mg/kg",
    "g/kg",
    "mg",
    "g",
    "kg",
    "mL/kg",
    "mL",
    "%",
    "ppm",
    "mg/g",
    # 영양 기준표 (S-043 · S-056)
    "IU",
    "IU/kg",
    "kcal",
    "kcal/kg",
    "kcal/g",
    "㎍",
    "µg",
    "ug",
    "kJ",
    "배",
    # 대사 체중 기준 에너지 요구량 — 개는 ^0.75, 고양이는 ^0.67 (S-043 p.25)
    "kcal/kg^0.75",
    "kcal/kg^0.67",
    "kJ/kg^0.75",
    "kJ/kg^0.67",
}

#: 경로① 스위치 (D-45). **이번 산출물 범위에서는 끈다.**
#:
#: 두 경로를 함께 켜면 같은 사실이 두 번 적재되어 top-k 를 잠식하고,
#: 원문의 완화 표현("an occasional apple seed will not harm")과
#: 사실 표의 `NEVER` 가 ④ 근거 검증에서 모순으로 판정될 수 있다.
#: 되돌릴 때는 이 값만 True 로 바꾼다 — 자격은 아래에서 원장이 정한다.
ROUTE1_ENABLED = False

#: 원문 복제를 허용하는 라이선스 표기 (01d §2.1 제출용 포함 기준).
_OPEN_LICENSE = ("CC BY", "정부", "보도자료", "공공누리", "public domain")


def route1_eligible() -> set[str]:
    """경로①(원문 청크 적재) 자격이 있는 `source_id` — **원장에서 유도한다.**

    하드코딩하지 않는 이유가 있다. 이전 판은 목록을 코드에 박아두었는데
    **8건 중 6건이 원장과 어긋나 있었다** — 삭제된 자료(`S-001`)가 남아 있었고,
    복제 금지(`B 가공활용`) 자료가 허용 목록에 들어 있었다.
    목록과 원장이 따로 놀면 어긋난 사실을 아무도 모른다 (D-22 단일 출처).

    자격 기준 세 가지를 **모두** 만족해야 한다.

    1. 원장이 배포 가능으로 표시했다 — 스냅샷은 `submit_ok=O`,
       원본은 `license` 가 오픈 라이선스·정부 간행물
    2. 품질이 `완전` 또는 `부분` 이다 — `요약` 을 원문으로 인용하면 그게 곧 환각이다 (D-30)
    3. 삭제 이력에 없다 (D-33 · G1a)

    원장을 읽지 못하면 **빈 집합**을 돌려준다. 자격이 없는 쪽으로 실패한다.
    """
    deleted: set[str] = set()
    dl = MANIFEST_DIR / "DELETION_LOG.csv"
    if dl.exists():
        rows = csv.DictReader(dl.open(encoding="utf-8-sig"))
        deleted = {(r.get("source_id") or "").strip() for r in rows}

    ok: set[str] = set()
    snap = MANIFEST_DIR / "SNAPSHOT_MANIFEST.csv"
    if snap.exists():
        for r in csv.DictReader(snap.open(encoding="utf-8-sig")):
            if (r.get("submit_ok") or "").strip() == "O" and (
                r.get("quality") or ""
            ).strip() in CITABLE_QUALITY:
                ok.add((r.get("source_id") or "").strip())

    raw = MANIFEST_DIR / "MANIFEST.csv"
    if raw.exists():
        for r in csv.DictReader(raw.open(encoding="utf-8-sig")):
            lic = r.get("license") or ""
            if any(k in lic for k in _OPEN_LICENSE):
                ok.add((r.get("source_id") or "").strip())

    return {s for s in ok if s and s not in deleted}


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


def check_row(row: dict[str, str], where: str, eligible: set[str] | None = None) -> list[Issue]:
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

    # `nutrition` 의 수치는 **권장량**이지 섭취 역치가 아니다.
    # 이 규칙은 중독 자료(toxicity_*)를 염두에 두고 만들었고, 영양 기준표에 그대로 걸면
    # 200행 넘는 권장량이 억지로 `기타` 를 달게 된다 — 뜻이 없는 값이 표에 쌓인다.
    is_nutrition = g("doc_type") == "nutrition"

    if dose and not ttype and not is_nutrition:
        # 성격을 모르는 수치는 역치로 오인된다. 지침 3장.
        out.append(
            Issue("ERROR", where, "수치가 있는데 threshold_type 이 비었다 — 지침 3장을 볼 것")
        )
    if is_nutrition and ttype in USABLE_THRESHOLDS:
        # 권장량이 규칙 테이블에 들어가면 "이 이상 먹으면 위험"으로 뒤집힌다.
        out.append(
            Issue(
                "ERROR",
                where,
                f"nutrition 인데 threshold_type={ttype!r} 이다 — "
                "권장량은 중독 역치가 아니다. 비우거나 '성분 함량'을 쓸 것",
            )
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

    # ── 원문(quote)을 실을 수 있는가 (D-37 · D-45) ───────────
    if g("quote"):
        if not ROUTE1_ENABLED:
            out.append(
                Issue(
                    "ERROR",
                    where,
                    "경로①(원문 적재)은 이번 범위에서 꺼져 있다 (D-45) — quote 를 비울 것",
                )
            )
        elif g("source_id") not in (eligible or set()):
            out.append(
                Issue(
                    "ERROR",
                    where,
                    f"{g('source_id')} 는 사실추출 한정 자료다 — quote 를 비울 것 (D-37)",
                )
            )

    # ── 단위 오식 ───────────────────────────────────────────
    if unit and unit.replace(" ", "") not in KNOWN_UNITS:
        out.append(Issue("WARN", where, f"보기 드문 단위: {unit!r} — 원문과 대조할 것"))

    return out


def check_file(
    path: Path, eligible: set[str] | None = None
) -> tuple[list[Issue], list[dict[str, str]]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    issues: list[Issue] = []
    for i, row in enumerate(rows, start=2):  # 헤더가 1행
        issues += check_row(row, f"{path.name}:{i}", eligible)
    return issues, rows


#: 검색에서 **함께 반환되는** 종끼리 묶는다 (D-39 병합 검색).
#:
#: `cat` 질의는 `cat`·`mammal`·`all` 을 함께 본다. 그러므로
#: `우유(cat)=NEVER` 와 `우유(mammal)=CAUTION` 은 **같은 답변 안에서 충돌한다.**
#: 종 문자열이 다르다는 이유로 상충 검사를 건너뛰면 그 모순이 그대로 적재된다.
CROSS_SPECIES_GROUPS: dict[str, str] = {
    "dog": "포유류",
    "cat": "포유류",
    "mammal": "포유류",
    "bird": "조류",
    "all": "*",  # 모든 그룹과 함께 나온다
}


def _cross_keys(species: str) -> tuple[str, ...]:
    g = CROSS_SPECIES_GROUPS.get(species.strip(), species.strip())
    return ("포유류", "조류") if g == "*" else (g,)


def check_cross(all_rows: list[dict[str, str]]) -> list[Issue]:
    """표 전체를 가로지르는 검사.

    ⚠️ 이 함수는 2026-08-02 까지 **한 번도 실행되지 않았다.**
    `main()` 이 `if len(paths) > 1:` 로 감싸고 있었는데 사실 표는 한 개뿐이라
    조건이 늘 거짓이었다. `make facts` 는 `ERROR 0` 으로 통과했고,
    그동안 사과·수박·복숭아·망고의 급여 등급 충돌이 그대로 적재되고 있었다.

        개에게 사과는 안전으로 분류된다. (출처: American Kennel Club, S-064)
        개에게 사과는 조건부 급여다. 1회 권장량은 …  (출처: Hill's, S-047)

    **파일 수와 무관하게 항상 돈다.** 한 파일 안에서도 출처 간 충돌은 일어난다.
    """
    out: list[Issue] = []

    dup = [k for k, v in Counter(r.get("fact_id", "") for r in all_rows).items() if v > 1 and k]
    for k in sorted(dup):
        out.append(Issue("ERROR", "병합", f"fact_id 중복: {k}"))

    # 같은 (물질 · 종그룹) 에 서로 다른 급여 등급이 붙으면 어느 쪽이 맞는지 정해야 한다.
    # 출처까지 함께 남긴다 — 무엇을 펼쳐 봐야 하는지 알려주지 않으면 검수가 안 된다.
    grades: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in all_rows:
        lv = (r.get("feeding_level") or "").strip()
        if not lv:
            continue
        for g in _cross_keys(r.get("species", "")):
            grades[(r.get("substance", ""), g)][lv].add(r.get("source_id", ""))
    for (sub, g), by_level in sorted(grades.items()):
        if len(by_level) > 1:
            detail = " vs ".join(
                f"{lv}({','.join(sorted(src))})" for lv, src in sorted(by_level.items())
            )
            out.append(
                Issue("WARN", "병합", f"{sub}[{g}] 급여 등급이 엇갈린다: {detail} — 검수 필요")
            )
    return out


def main() -> int:
    args = sys.argv[1:]
    paths = [Path(a) for a in args] if args else sorted(FACTS_DIR.glob("facts_*.csv"))
    if not paths:
        print(f"검사할 파일이 없다. {FACTS_DIR} 에 facts_*.csv 를 만들 것.")
        return 1

    print("사실 표 검사 (01e 지침)\n")
    eligible = route1_eligible()
    if ROUTE1_ENABLED:
        print(f"  · 경로① 켜짐 — 자격 {len(eligible)}건: {', '.join(sorted(eligible))}\n")
    issues: list[Issue] = []
    all_rows: list[dict[str, str]] = []

    for p in paths:
        if not p.exists():
            print(f"  ✗ 파일 없음: {p}")
            return 1
        file_issues, rows = check_file(p, eligible)
        all_rows += rows
        issues += file_issues
        print(f"[{p.name}]  {len(rows)}행")
        for it in file_issues:
            print(it)
        if not file_issues:
            print("  · 문제 없음")
        print()

    # **파일 수와 무관하게 항상 돈다.** 예전에는 `if len(paths) > 1:` 이었고,
    # 사실 표가 한 개라 이 검사가 한 번도 실행되지 않았다 (2026-08-02 검토).
    cross = check_cross(all_rows)
    issues += cross
    print("[표 전체]")
    for it in cross:
        print(it)
    if not cross:
        print("  · 문제 없음")
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
