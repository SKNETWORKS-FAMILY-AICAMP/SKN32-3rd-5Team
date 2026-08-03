#!/usr/bin/env python3
"""리포트 JSON 두 개를 **케이스 단위로** 비교한다 (04 §7).

    python scripts/diff_reports.py eval/reports/A.json eval/reports/B.json

## 왜 필요한가

    2026-08-03 — `통과 45.0% → 43.3%` 로 1건 떨어졌는데 `등급 일치도`는
    `90.0% → 93.1%` 로 올랐다. 좋아진 것처럼 보이지만 **틀리던 건이 거절로 빠져
    등급 분모에서 사라진 것**이었다 (30 → 29). 04 §1.2 가 경고한
    *"답을 안 한 것이 만점으로 보고된다"* 가 집계 사이에서 일어난다.

    **총계만 보면 이 방향을 못 읽는다.** 어느 케이스가 어디로 갔는지를 봐야 한다.

## 무엇을 내는가

    통과→실패 / 실패→통과 로 **뒤집힌 케이스**와, 그 안에서 무엇이 달라졌는지
    (상태 · 등급 · 게이트 · 근거) 를 한 줄씩. 바뀐 것이 없으면 그렇게 말한다.

⚠️ **모델이 같아도 결과는 흔들릴 수 있다.** `temperature=0` 은 박혀 있으나
   `seed` 를 주지 않으므로 OpenAI 는 재현을 보장하지 않는다 (04 §8).
   따라서 *"이 차이가 내 변경 때문인가"* 는 이 도구만으로 단정할 수 없다 —
   **같은 코드로 두 번 돌려 흔들림의 크기를 먼저 재야 한다.**
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: 케이스별로 비교할 칸. **이름을 여기 모아 둔다** — 늘어날 때 한 곳만 고친다.
FIELDS = (
    ("actual_status", "상태"),
    ("actual_level", "등급"),
    ("rule_level", "rule"),
    ("llm_level", "llm"),
    ("gate_capped", "상승차단"),
    ("gate_overridden", "하향차단"),
    ("actual_refusal_reason", "거절이유"),
    ("cite_any", "cite"),
    ("contain_ok", "contain"),
    ("not_contain_ok", "금지어"),
    ("grounding_unsupported", "근거없음"),
)


def _load(p: Path) -> tuple[dict[str, dict], dict[str, Any]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    return {c["case_id"]: c for c in data.get("cases", [])}, data


def _summary_line(name: str, data: dict[str, Any], cases: dict[str, dict]) -> str:
    passed = sum(1 for c in cases.values() if c.get("passed"))
    graded = sum(1 for c in cases.values() if c.get("level_delta") is not None)
    return (
        f"  {name:22} 통과 {passed:>3}/{len(cases)}   등급분모 {graded:>3}   "
        f"모델 {data.get('model', '?')}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    a = ap.parse_args()

    old, old_meta = _load(a.before)
    new, new_meta = _load(a.after)

    print("=" * 72)
    print(_summary_line(a.before.name, old_meta, old))
    print(_summary_line(a.after.name, new_meta, new))
    print("=" * 72)

    # 🔴 **분모가 달라졌으면 비율을 나란히 읽으면 안 된다.** 먼저 말한다.
    g_old = sum(1 for c in old.values() if c.get("level_delta") is not None)
    g_new = sum(1 for c in new.values() if c.get("level_delta") is not None)
    if g_old != g_new:
        print(
            f"\n⚠️ **등급 분모가 {g_old} → {g_new} 로 바뀌었다.** 등급 일치도·과소·과대를\n"
            "   두 실행 사이에서 그대로 비교하면 안 된다 — 분모가 줄면 비율이 오른다 (04 §1.2)."
        )

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    if only_old or only_new:
        print(f"\n⚠️ 한쪽에만 있는 케이스 — before {only_old} / after {only_new}")

    broke, fixed, changed = [], [], []
    for cid in sorted(set(old) & set(new)):
        o, n = old[cid], new[cid]
        diffs = [
            f"{label} {o.get(key)!r}→{n.get(key)!r}"
            for key, label in FIELDS
            if o.get(key) != n.get(key)
        ]
        if o.get("passed") and not n.get("passed"):
            broke.append((cid, o, diffs))
        elif not o.get("passed") and n.get("passed"):
            fixed.append((cid, o, diffs))
        elif diffs:
            changed.append((cid, o, diffs))

    def _show(title: str, rows: list, mark: str) -> None:
        print(f"\n■ {title} {len(rows)}건")
        if not rows:
            print("  없음")
            return
        for cid, o, diffs in rows:
            head = f"  {mark} {cid:8} {o.get('case_type', ''):8} {o.get('species', ''):6}"
            print(head + ("  " + " · ".join(diffs) if diffs else "  (채점 칸은 그대로)"))

    _show("🔴 통과 → 실패", broke, "✗")
    _show("✅ 실패 → 통과", fixed, "✓")
    _show("통과 여부는 같으나 내용이 달라진 것", changed, "·")

    if not (broke or fixed or changed):
        print("\n두 실행이 케이스 단위로 완전히 같다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
