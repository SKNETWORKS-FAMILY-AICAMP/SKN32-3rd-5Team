#!/usr/bin/env python
"""골든셋 검사 — 커밋 전에 돌린다.

    python scripts/check_goldenset.py eval/goldenset/golden_lse.csv
    python scripts/check_goldenset.py                # golden_*.csv 전부

설계 근거: docs/04a_골든셋작성지침.md · docs/04 §2.3 · docs/06 D-13 · D-39

**골든셋이 틀리면 평가 전체가 틀린다.** 정답이 잘못된 채로 지표를 내면
그 지표는 아무 의미가 없으므로, 적재 전에 형식과 논리 모순을 거른다.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "eval" / "goldenset"
MANIFEST_DIR = ROOT / "data" / "manifests"


def known_source_ids() -> set[str]:
    """코퍼스에 실재하는 `source_id` 집합.

    **세 대장의 합집합**을 쓴다. 한 곳만 보면 멀쩡한 골든셋을 오류로 잡는다.

    | 대장 | 무엇이 있나 |
    |---|---|
    | `SOURCES_CITED.csv` | 인용 대장 — 제출물에 나가는 출처 목록 |
    | `SNAPSHOT_MANIFEST.csv` | 스냅샷 42건 |
    | `MANIFEST.csv` | 원본 보유 6건 + 내부 산출물 |

    `SNAPSHOT_MANIFEST` 를 빠뜨렸다가 **S-016·S-029·S-034·S-043~046·S-070 8건**이
    "대장에 없다"로 잘못 잡혔다. 공교롭게 그 8건이 원문 배포가 가능한 자료들이라
    인용 대장에 행을 안 만들어 둔 것이었다 (01d §2.2).

    대장을 못 읽으면 빈 집합을 돌려주고 검사를 건너뛴다 (04 §8: 검사 축소는 드러나야 한다).
    """
    ids: set[str] = set()
    for name in ("SOURCES_CITED.csv", "SNAPSHOT_MANIFEST.csv", "MANIFEST.csv"):
        p = MANIFEST_DIR / name
        if not p.exists():
            continue
        for row in csv.DictReader(p.open(encoding="utf-8-sig")):
            sid = (row.get("source_id") or "").strip()
            if sid:
                ids.add(sid)
    return ids


REQUIRED = ("case_id", "question", "expected_status", "difficulty")
STATUSES = {"answered", "clarify", "refused"}
TRIAGE = {"EMERGENCY", "CALL_NOW", "VISIT_SOON", "MONITOR"}
SPECIES = {"dog", "cat", "bird", ""}
INTENTS = {"intoxication", "symptom", "nutrition", "general", ""}
REFUSAL = {"근거없음", "검증실패", "되묻기상한", "판정불가", "범위밖", ""}
DIFFICULTY = {"쉬움", "보통", "어려움"}

#: 종별 최소 건수 (04 §2.3). 한쪽으로 쏠리면 종별 지표를 낼 수 없다.
MIN_PER_SPECIES = {"dog": 10, "cat": 10, "bird": 10}
#: 복사해서 쓰는 원본. 여기에 직접 쓰지 않으므로 총량 기준에서 제외한다.
TEMPLATE_NAME = "골든셋_양식.csv"
#: 상태별 최소 비율 (04 §2.2). 거절·되묻기가 없으면 그 경로를 평가할 수 없다.
#: `clarify` 10% = 슬롯 결측형, `refused` 15% = 답 없음.
MIN_RATIO = {"clarify": 0.10, "refused": 0.15}


@dataclass
class Issue:
    level: str
    where: str
    message: str

    def __str__(self) -> str:
        return f"  {'✗' if self.level == 'ERROR' else '⚠'} [{self.where}] {self.message}"


def check_row(r: dict[str, str], where: str, known: set[str] | None = None) -> list[Issue]:
    out: list[Issue] = []
    g = lambda k: (r.get(k) or "").strip()  # noqa: E731

    for f in REQUIRED:
        if not g(f):
            out.append(Issue("ERROR", where, f"필수 칸이 비었다: {f}"))

    st = g("expected_status")
    if st and st not in STATUSES:
        out.append(Issue("ERROR", where, f"expected_status 오타: {st!r} — {sorted(STATUSES)}"))
    if g("species") not in SPECIES:
        out.append(Issue("ERROR", where, f"species 오타: {g('species')!r}"))
    if g("intent") not in INTENTS:
        out.append(Issue("ERROR", where, f"intent 오타: {g('intent')!r}"))
    if g("difficulty") and g("difficulty") not in DIFFICULTY:
        out.append(
            Issue("ERROR", where, f"difficulty 오타: {g('difficulty')!r} — {sorted(DIFFICULTY)}")
        )

    tri = g("expected_triage")
    if tri and tri not in TRIAGE:
        out.append(Issue("ERROR", where, f"expected_triage 오타: {tri!r}"))

    # ── 상태별 논리 정합성 ──────────────────────────────────
    if st == "answered":
        if not g("must_cite"):
            out.append(
                Issue(
                    "WARN",
                    where,
                    "answered 인데 must_cite 가 비었다 — 근거 없는 정답은 채점이 안 된다",
                )
            )
        if g("expected_refusal_reason"):
            out.append(Issue("ERROR", where, "answered 인데 거절 사유가 적혀 있다"))
    elif st == "refused":
        if not g("expected_refusal_reason"):
            out.append(Issue("ERROR", where, "refused 인데 사유가 없다 — 오류 분석에 쓸 수 없다"))
        if tri:
            out.append(Issue("ERROR", where, "refused 인데 트리아지 등급이 적혀 있다"))
    elif st == "clarify" and tri:
        out.append(Issue("ERROR", where, "clarify 인데 트리아지 등급이 적혀 있다"))

    if g("expected_refusal_reason") not in REFUSAL:
        out.append(Issue("ERROR", where, f"거절 사유 오타: {g('expected_refusal_reason')!r}"))

    # ── must_cite 가 실재하는 자료인가 ──────────────────────
    # 없는 `source_id` 를 정답에 적으면 그 케이스는 **영원히 통과할 수 없다.**
    for sid in [s.strip() for s in g("must_cite").split("|") if s.strip()]:
        if known and sid not in known:
            out.append(
                Issue(
                    "ERROR",
                    where,
                    f"must_cite 의 {sid} 가 대장에 없다 — 오타이거나 미수집 자료다",
                )
            )

    # ── 종 관련 안전 조건 ───────────────────────────────────
    if g("species") == "bird" and any(k in g("question") for k in ("체중당", "mg/kg", "g/kg")):
        out.append(
            Issue("WARN", where, "조류 질문에 체중당 수치가 있다 — 코퍼스에 조류 임계치는 0건이다")
        )
    if st == "clarify" and g("species") and "체중" not in g("must_contain"):
        pass  # 종 되묻기일 수도 있다. 강제하지 않는다

    # ── 증상 질의가 원인을 지목하게 두지 않는다 (D-11 · D-49) ──
    #
    # 코퍼스는 물질 단위라 각 청크가 "이 물질 → 이런 증상" 이다. 증상만 주면
    # 그 화살표를 거꾸로 타는데 **역방향은 일대일이 아니다** —
    # 고양이 청크 418건 중 21건이 같은 증상 조합(구토·식욕부진·침흘림·복통)을 나열한다.
    #
    # 실제로 '고양이가 토하고 밥을 안 먹고 배를 아파해요' 의 1위가 **토마토**였다.
    # 검색은 맞게 일했다 — 증상이 실제로 겹친다. 그걸 근거로 답하면
    # *"토마토 중독일 수 있습니다"* 가 되고, **증상에서 원인을 지목하는 것이 곧 진단이다.**
    #
    # 증상 질의에 `answered` 자체는 정당하다 — 물질을 몰라도 **등급은 말할 수 있다** (D-39).
    # 다만 그 정답에는 **무엇을 말하면 안 되는지**가 적혀 있어야 채점이 성립한다.
    if g("intent") == "symptom" and st == "answered" and not g("must_not_contain"):
        out.append(
            Issue(
                "WARN",
                where,
                "증상 질의인데 must_not_contain 이 비었다 — "
                "증상만으로 원인 물질·질환을 지목하면 진단이다 (D-11 · D-49). "
                "금지 문구를 적어야 채점된다",
            )
        )

    if tri == "MONITOR" and "연락" not in g("must_contain") and "증상" not in g("must_contain"):
        out.append(
            Issue("WARN", where, "MONITOR 정답인데 상승 조건 문구가 must_contain 에 없다 (D-39)")
        )
    return out


def check_coverage(rows: list[dict[str, str]], *, quota: bool = True) -> list[Issue]:
    """`quota=False` 면 총량 목표(종별 건수·상태 비율)를 건너뛴다.

    양식 파일 하나만 검사할 때는 100건 기준이 의미가 없다.
    행 단위 검사는 그대로 돈다 — 양식이 지침을 어기면 그건 잡아야 한다.
    """
    out: list[Issue] = []
    n = len(rows)
    if not n:
        return out

    if quota:
        sp = Counter((r.get("species") or "").strip() for r in rows)
        for s, need in MIN_PER_SPECIES.items():
            if sp.get(s, 0) < need:
                out.append(
                    Issue("WARN", "커버리지", f"{s} {sp.get(s, 0)}건 — 목표 {need}건 (04 §2.3)")
                )

        stat = Counter((r.get("expected_status") or "").strip() for r in rows)
        for s, ratio in MIN_RATIO.items():
            got = stat.get(s, 0) / n
            if got < ratio:
                out.append(
                    Issue(
                        "WARN",
                        "커버리지",
                        f"{s} {stat.get(s, 0)}건 ({got:.0%}) — 목표 {ratio:.0%}. "
                        "이 경로를 평가할 수 없다",
                    )
                )

    dup = [k for k, v in Counter(r.get("case_id", "") for r in rows).items() if v > 1 and k]
    for k in sorted(dup):
        out.append(Issue("ERROR", "병합", f"case_id 중복: {k}"))

    seen: dict[str, str] = {}
    for r in rows:
        q = (r.get("question") or "").strip()
        if q in seen:
            out.append(Issue("WARN", "병합", f"질문 중복: {r.get('case_id')} ↔ {seen[q]}"))
        seen[q] = r.get("case_id", "")

    # 검수자 미기입
    unrev = [r.get("case_id") for r in rows if not (r.get("reviewed_by") or "").strip()]
    if unrev:
        out.append(
            Issue(
                "WARN",
                "검수",
                f"검수자 미기입 {len(unrev)}건 — 작성자·검수자는 분리한다 (04 §2.4)",
            )
        )
    return out


def main() -> int:
    args = sys.argv[1:]
    paths = [Path(a) for a in args] if args else sorted(GOLDEN_DIR.glob("golden_*.csv"))
    if not paths:
        print(f"검사할 파일이 없다. {GOLDEN_DIR} 에 golden_*.csv 를 만들 것.")
        print("양식: eval/goldenset/골든셋_양식.csv")
        return 1

    print("골든셋 검사 (04a 지침)\n")
    known = known_source_ids()
    if not known:
        print("  ⚠ 대장을 읽지 못했다 — must_cite 실재 검사를 건너뛴다 (04 §8)\n")
    issues: list[Issue] = []
    rows: list[dict[str, str]] = []
    for p in paths:
        if not p.exists():
            print(f"  ✗ 파일 없음: {p}")
            return 1
        file_rows = list(csv.DictReader(p.open(encoding="utf-8-sig")))
        rows += file_rows
        file_issues = [
            i
            for n, r in enumerate(file_rows, start=2)
            for i in check_row(r, f"{p.name}:{n}", known)
        ]
        issues += file_issues
        print(f"[{p.name}]  {len(file_rows)}건")
        for i in file_issues:
            print(i)
        if not file_issues:
            print("  · 문제 없음")
        print()

    # 양식 하나만 검사할 때는 100건 총량 기준을 적용하지 않는다.
    quota = not (len(paths) == 1 and paths[0].name == TEMPLATE_NAME)
    cov = check_coverage(rows, quota=quota)
    issues += cov
    print("[커버리지]" if quota else "[커버리지] (양식 단독 검사 — 총량 기준 생략)")
    for i in cov:
        print(i)
    if not cov:
        print("  · 문제 없음")
    print()

    by_status = Counter((r.get("expected_status") or "?").strip() for r in rows)
    by_species = Counter((r.get("species") or "(미지정)").strip() for r in rows)
    errors = sum(1 for i in issues if i.level == "ERROR")
    warns = sum(1 for i in issues if i.level == "WARN")
    print(f"→ 총 {len(rows)}건 · 상태 {dict(by_status)} · 종 {dict(by_species)}")
    print(f"→ ERROR {errors} · WARN {warns}")
    if errors:
        print("\n  ERROR 를 고치고 커밋한다. 지침: docs/04a_골든셋작성지침.md")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
