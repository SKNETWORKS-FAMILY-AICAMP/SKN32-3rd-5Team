#!/usr/bin/env python
"""사실 표 → **물질 어휘 폐쇄 목록.**

    python scripts/build_vocabulary.py          # 미리보기
    python scripts/build_vocabulary.py --write  # compute/tables/ 에 기록

설계 근거: docs/06 D-59 ① · D-22 · D-38 · D-40

**D-59 ①이 정한 것을 파일로 만든다.**

    ✗  "이 상황의 물질은?"                   → 생성 → 환각 가능
    ✓  "다음 중 어느 것인가? 없으면 '없음'"    → 선택 → 환각 불가능

그 *"다음"* 이 이 파일이다. 프롬프트가 목록을 말로 설명하는 것과
**목록이 파일로 존재하고 계약이 그것을 강제하는 것**은 다르다 (D-40).

왜 코퍼스를 런타임에 안 읽고 표로 뽑아 두나
------------------------------------------
`data/facts/facts_*.csv` 는 **패키지에 안 들어간다.** 설치본에는 `compute/tables/`
만 실린다. 어휘를 런타임에 코퍼스에서 만들면 설치 환경에서 어휘가 **조용히 비고**,
그러면 폐쇄 목록 검사가 전부 통과하거나 전부 실패한다. 둘 다 나쁘다.

`정량임계치.csv` 와 같은 방식으로 **생성물을 파일로 고정**한다.
사람이 고치지 않는다 — 고칠 곳은 `facts_*.csv` 다 (D-22).

`make vocab` 으로 재생성하고, `tests/test_vocabulary.py` 가
**표가 코퍼스와 어긋나면** 잡는다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pettriage import paths  # noqa: E402
from pettriage.ingest.facts_io import load_all  # noqa: E402

OUT_NAME = "물질어휘.csv"
FIELDS = ("substance", "species", "n_facts")


def build(facts) -> list[dict[str, str]]:
    """(물질명 → 종 집합 · 사실 건수). **물질명은 코퍼스에 적힌 그대로다.**

    정규화하지 않는다. `알리움류(양파·마늘·리크·차이브)` 를 `알리움류` 로 줄이면
    **우리가 만든 이름**이 되고, 그 순간 폐쇄 목록이 아니라 생성물이 된다 (D-38).
    보호자 어휘와의 다리는 `별칭.csv` 가 놓는다 — 거기는 근거가 붙는다.
    """
    species: dict[str, set[str]] = defaultdict(set)
    count: dict[str, int] = defaultdict(int)
    for f in facts:
        name = (f.substance or "").strip()
        if not name:
            continue
        species[name].add(f.species)
        count[name] += 1
    return [
        {
            "substance": name,
            "species": "|".join(sorted(species[name])),
            "n_facts": str(count[name]),
        }
        for name in sorted(species)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="사실 표에서 물질 어휘 폐쇄 목록을 뽑는다")
    ap.add_argument("--facts-dir", type=Path, default=None)
    ap.add_argument("--write", action="store_true", help="compute/tables/ 에 기록한다")
    args = ap.parse_args()

    root = paths.find_root() or Path.cwd()
    facts_dir = args.facts_dir or root / "data" / "facts"
    if not facts_dir.is_dir():
        print(f"✗ 사실 표 폴더가 없다: {facts_dir}")
        return 1

    facts = load_all(facts_dir)
    if not facts:
        print(f"✗ {facts_dir} 에 facts_*.csv 가 없다")
        return 1

    rows = build(facts)
    print(f"사실 {len(facts)}행 → 물질 {len(rows)}종")
    per_species: dict[str, int] = defaultdict(int)
    for r in rows:
        for s in r["species"].split("|"):
            per_species[s] += 1
    print("  종별(중복 포함): " + " · ".join(f"{k} {v}" for k, v in sorted(per_species.items())))

    if not args.write:
        print("\n예시 5종")
        for r in rows[:5]:
            print(f"  {r['substance']}  [{r['species']}]  {r['n_facts']}건")
        print(f"\n기록하려면 --write  (→ compute/tables/{OUT_NAME})")
        return 0

    out = root / "src" / "pettriage" / "compute" / "tables" / OUT_NAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\n→ {out}  ({len(rows)}행)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
