#!/usr/bin/env python3
"""② 슬롯이 **무엇을 뽑았는지 눈으로 본다** (D-85 후속 진단).

    python scripts/probe_slots.py --only G-020,G-026,G-049,G-050,G-051,G-052
    python scripts/probe_slots.py --type none

## 왜 필요한가

    2026-08-03 D-85(*"말했는데 코퍼스에 없으면 되묻지 않고 근거없음"*)를 넣었는데
    여섯 건 중 **한 건만** 통과했다. 다섯 건은 그대로 되묻기로 갔다.

    코드 경로를 아무리 읽어도 이유를 못 좁힌다. `unknown_substance` 가 서려면
    `surface` 가 있어야 하는데, **그 값이 무엇인지 리포트 JSON 에 없다.**
    슬롯은 응답에 `clarify.missing` 로만 나가고 표면형은 어디에도 안 실린다.

    *"이 상태를 실제로 만들 수 있는가"* 를 물으려면 그 상태를 봐야 한다.

## 무엇을 하지 않는가

    **고치지 않는다.** 채점도 안 한다. `extract_slots` 를 한 번 태우고
    상태에 남은 것을 그대로 찍는다.

    ⚠️ ②슬롯 LLM 을 부른다 (건당 1회, 5건 ≈ 10초).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GOLDEN_DIR = ROOT / "eval" / "goldenset"


def _rows() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for p in sorted(GOLDEN_DIR.glob("golden_*.csv")):
        with p.open(encoding="utf-8-sig", newline="") as f:
            out += [{k: (v or "").strip() for k, v in r.items()} for r in csv.DictReader(f)]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="case_id 를 쉼표로")
    ap.add_argument("--type", help="case_type 으로 거른다")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument(
        "--arm",
        choices=["none", "A", "A-LC", "C", "D"],
        help="비교군을 고정한다. 없으면 셸·.env 를 그대로 탄다",
    )
    a = ap.parse_args(argv)

    # ⚠️ 설정을 읽기 전에 세운다 — `get_config` 는 `lru_cache` 다.
    if a.arm:
        from pettriage.models.serving.arms import apply_arm

        print(f"비교군 {a.arm} — {apply_arm(a.arm)}")

    rows = _rows()
    if a.only:
        want = {s.strip() for s in a.only.split(",") if s.strip()}
        rows = [r for r in rows if r.get("case_id") in want]
    if a.type:
        rows = [r for r in rows if r.get("case_type") == a.type]
    rows = rows[: a.n]
    if not rows:
        print("해당하는 케이스가 없다.")
        return 1

    from pettriage.compute.vocabulary import mention_in, resolve_substance
    from pettriage.graph.nodes.slots import _is_vague, _llm_slots, extract_slots

    print("=" * 86)
    print(f"  ② 슬롯 표면형 진단 — {len(rows)}건")
    print("=" * 86)

    for r in rows:
        q = r["question"]
        sp_gold = r.get("species") or None
        print(f"\n{'─' * 86}")
        print(f"  {r['case_id']}  [{r.get('case_type')}·{r.get('species')}]  «{q}»")

        # ① LLM 이 낸 날것
        raw = _llm_slots(q)
        llm_surface = (raw or {}).get("substance")
        print(f"    LLM 원본        : {raw}")

        # ② 폴백이 잡는 것
        fb = mention_in(q, sp_gold)
        print(f"    폴백(mention_in): {fb!r}")

        surface = llm_surface or fb
        print(f"    → surface       : {surface!r}   vague={_is_vague(surface)}")

        if surface:
            res_sp = resolve_substance(surface, sp_gold)
            res_all = resolve_substance(surface, None)
            print(
                f"    resolve(종O)    : name={res_sp.name!r} how={res_sp.how!r} "
                f"후보={len(res_sp.candidates)}"
            )
            print(f"    resolve(종X)    : name={res_all.name!r}")

        # ③ 노드가 실제로 낸 것
        st = {"question": q, "slots": {}, "intent": r.get("intent", "")}
        out = extract_slots(st)  # type: ignore[arg-type]
        keys = ("substance_candidates", "off_species_substance", "unknown_substance")
        marks = {k: out.get(k) for k in keys if out.get(k)}  # type: ignore[union-attr]
        print(f"    slots           : {out.get('slots')}")
        print(f"    missing_slots   : {out.get('missing_slots')}")
        print(f"    갈래            : {marks or '(없음 — 되묻기로 간다)'}")

    print(f"\n{'=' * 86}")
    print("  읽는 법 — `unknown_substance` 가 서야 D-85 가 발동한다.")
    print("  안 섰다면 그 앞 세 줄 중 하나에서 갈렸다: surface 없음 / 모호어 / 후보·종밖.")
    print("=" * 86)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
