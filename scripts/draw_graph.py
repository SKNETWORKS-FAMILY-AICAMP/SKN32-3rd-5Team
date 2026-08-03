#!/usr/bin/env python3
"""질의 그래프 구조도를 **코드에서** 뽑는다 (필수 산출물 ② 시스템 아키텍처).

    python scripts/draw_graph.py            # docs/그림/질의그래프.mmd 갱신
    python scripts/draw_graph.py --stdout   # 화면으로만

⚠️ **손으로 그린 그림을 문서에 붙이지 않는다.**

    2026-08-02 까지 `graph/nodes/__init__.py` 머리말에 아스키 구조도가 있었는데,
    거기 `finalize` 가 *"모든 경로가 마지막에 만난다"* 로 그려져 있었다.
    실제로는 `_run_pipeline` 이 `finalize` 를 **한 번도 부르지 않았다** —
    연락처 차단은 `SafetyEngine` 래퍼로 옮겨 갔고 그림만 남아 있었다.

    손으로 그린 그림은 코드가 바뀌어도 안 바뀐다. **틀린 그림은 없는 그림보다 나쁘다** —
    읽는 사람이 그것을 믿고 코드를 안 보기 때문이다. 이 스크립트가 그리는 것은
    `build_graph()` 가 실제로 조립한 엣지뿐이라 **틀릴 수가 없다** (D-22 · D-38).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "docs" / "그림" / "질의그래프.mmd"

HEADER = """%% 이 파일은 `python scripts/draw_graph.py` 가 생성한다. 손으로 고치지 않는다.
%% 원본은 src/pettriage/graph/build.py 의 build_graph() 다.
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true", help="파일에 쓰지 않고 화면으로만")
    args = ap.parse_args()

    from pettriage.graph.build import get_graph

    mermaid = get_graph().get_graph().draw_mermaid()
    text = HEADER + mermaid

    if args.stdout:
        print(text)
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(f"✅ {OUT.relative_to(ROOT)} ({len(mermaid.splitlines())}줄)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
