#!/usr/bin/env python
"""사실 표 → 트리아지 규칙 테이블.

    python scripts/build_rule_table.py          # 미리보기
    python scripts/build_rule_table.py --write  # compute/tables/ 에 기록

설계 근거: docs/06 D-16 · D-22 · D-38 · D-39

**규칙 테이블을 손으로 관리하지 않는다.**

    이전에는 `compute/tables/사실표_초안_정량임계치.csv` 를 사람이 따로 유지했고,
    그 결과 마카다미아 0.7 g/kg 이 `임상징후 발현` 으로 잘못 적혀 있었다 —
    원문은 *"the dose required to induce toxicity has not been established precisely"* 다.
    사실 표에서는 고쳤는데 초안 CSV는 그대로여서 **어느 쪽이 진짜인지 알 수 없는 상태**가 됐다.

    D-22의 단일 출처 원칙을 데이터에도 적용한다 — `facts_*.csv` 하나만 사람이 쓰고,
    규칙 테이블은 여기서 **파생**한다.

편입 기준은 `templates.THRESHOLD_TYPES` 와 **같은 집합**을 쓴다.
문장에 "N 이상 섭취 시" 로 나가는 것과 규칙 테이블에 들어가는 것이 어긋나면 안 된다.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pettriage import paths  # noqa: E402
from pettriage.compute.rules import COMPUTABLE_UNITS as COMPUTABLE_UNITS_FROM_RULES  # noqa: E402
from pettriage.ingest.templates import THRESHOLD_TYPES  # noqa: E402

OUT_NAME = "정량임계치.csv"

#: 계산 노드가 **체중과 곱해 판정할 수 있는** 단위.
#:
#: `seeds`·`leaves`·`drupes` 같은 개수 단위는 여기 없다 —
#: 원문이 개수로만 말했으므로 체중당으로 환산할 방법이 없다.
#: 그 행도 테이블에는 넣되 `computable=N` 으로 표시해 **계산 노드가 건너뛰게** 한다.
#:
#: ⚠️ **여기서 목록을 정하지 않는다.** 표를 읽는 쪽(`compute.rules`)의 환산표에서
#: 가져온다. 예전에는 각자 목록을 들고 있었고 실제로 어긋나 있었다 —
#: 이 파일은 `mL/kg` 을 "계산 가능"으로 표시했는데 `rules._MG_PER_KG` 에는 없어서,
#: 그 행이 조용히 계산에서 빠지거나 `min()` 이 빈 시퀀스로 터졌다 (2026-08-02 검토).
#: **쓰는 쪽과 읽는 쪽이 같은 목록을 보게** 만드는 것이 유일한 항구적 해결이다.
COMPUTABLE_UNITS = COMPUTABLE_UNITS_FROM_RULES

FIELDS = (
    "fact_id",
    "substance",
    "species",
    "threshold_type",
    "dose",
    "unit",
    "computable",
    "effect",
    "signs",
    "onset",
    "source_id",
    "citation",
    "note",
)


def is_computable(unit: str, dose: str) -> bool:
    """체중당 계산이 가능한가.

    `%` 는 체중 대비 백분율이라 계산된다 (알리움 0.5% · 란타나 ≥1%).
    범위(`15-30`)와 부등호(`≥1`)는 계산 노드가 **안전한 쪽(낮은 값)** 으로 읽는다.
    """
    if unit.strip() not in COMPUTABLE_UNITS:
        return False
    return bool(re.search(r"\d", dose or ""))


def build(facts_dir: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for p in sorted(facts_dir.glob("facts_*.csv")):
        for r in csv.DictReader(p.open(encoding="utf-8-sig")):
            if (r.get("threshold_type") or "").strip() not in THRESHOLD_TYPES:
                continue
            if not (r.get("dose") or "").strip():
                continue
            out.append(
                {
                    "fact_id": r["fact_id"],
                    "substance": r["substance"],
                    "species": r["species"],
                    "threshold_type": r["threshold_type"],
                    "dose": r["dose"],
                    "unit": r["unit"],
                    "computable": "Y" if is_computable(r["unit"], r["dose"]) else "N",
                    "effect": r.get("effect", ""),
                    # 역치 **미만**일 때 MONITOR 의 상승 조건으로 쓴다 (D-50).
                    # 없으면 `apply_gate` 가 MonitorWithoutConditions 를 던지고
                    # 부르는 쪽이 거절로 바꾼다 — **역치를 안 넘긴 질의가 전부 거절이 된다.**
                    "signs": r.get("signs", ""),
                    "onset": r.get("onset", ""),
                    "source_id": r["source_id"],
                    "citation": r.get("citation", ""),
                    "note": (r.get("note") or "")[:200],
                }
            )
    out.sort(key=lambda r: (r["substance"], r["species"], r["fact_id"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="사실 표에서 규칙 테이블을 파생한다")
    ap.add_argument("--write", action="store_true", help="compute/tables/ 에 기록")
    args = ap.parse_args()

    root = paths.find_root() or Path.cwd()
    rows = build(root / "data" / "facts")
    if not rows:
        print("✗ 편입 가능한 행이 없다. 사실 표의 threshold_type 을 확인할 것 (01e §3)")
        return 1

    yes = [r for r in rows if r["computable"] == "Y"]
    no = [r for r in rows if r["computable"] == "N"]
    print(f"규칙 테이블 {len(rows)}행  (편입 기준: {sorted(THRESHOLD_TYPES)})\n")
    for r in rows:
        mark = " " if r["computable"] == "Y" else "⚠"
        print(
            f"  {mark} {r['fact_id']:12} {r['substance'][:24]:26} {r['species']:7} "
            f"{r['dose']:>8} {r['unit']:<14} {r['threshold_type']}"
        )

    print(f"\n  계산 가능 {len(yes)}행 · **계산 불가 {len(no)}행**")
    if no:
        print("  계산 불가 행은 단위가 개수·비정형이라 체중당 판정을 할 수 없다.")
        print("  계산 노드는 이 행들을 건너뛰고 **정성 문장으로만** 답해야 한다.")
        for r in no:
            print(f"      {r['fact_id']} {r['substance'][:20]} — {r['dose']}{r['unit']}")

    if not args.write:
        print("\n기록하려면 --write")
        return 0

    dest = root / "src" / "pettriage" / "compute" / "tables" / OUT_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(FIELDS))
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {dest}")
    print("  ⚠ 이 파일은 **생성물이다.** 손으로 고치지 말고 사실 표를 고친 뒤 다시 돌린다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
