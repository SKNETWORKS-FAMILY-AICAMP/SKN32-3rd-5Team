#!/usr/bin/env python3
"""어휘표가 **이 말을 어떻게 읽는지** 본다. LLM 을 부르지 않는다 (즉시).

    python scripts/probe_vocab.py 목캔디 계피가루 "세제 거품" "니코틴 껌 조각"
    python scripts/probe_vocab.py --unknown        # D-85 가 거절로 보내는 여섯 개

## 왜 필요한가

    2026-08-03 D-87 사고. 표면형이 구(句)일 때 그 안의 어휘를 찾으려고
    `mention_in(surface, species)` 를 썼는데, 그 함수는 표면형을 **문장처럼**
    훑는다. `'목캔디'` 안에서도 무언가를 찾아내 `substance` 가 잘못 서고,
    D-85 의 `근거없음` 거절이 파이프라인 끝까지 흘러가 `판정불가` 가 됐다.
    통과 33 → 25.

    **그때 테스트에 `assert mention_in("목캔디","dog") is None` 이라고 적어 뒀는데
    한 번도 실행하지 않았다.** 검증 안 한 가정을 테스트로 적으면 그 테스트가
    가정을 사실로 보이게 만든다 — 이 스크립트는 그 한 줄을 3초에 확인한다.

## 읽는 법

    mention_in   문장에서 어휘를 찾는다. **표면형을 넣으면 부분 매칭이 일어난다**
    resolve      폐쇄 목록 위로 올린다. `name` 이 서면 슬롯에 들어간다
    후보         여럿이면 `모호` — 되묻지 않고 전부 검색어로 넘긴다 (D-62)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: D-85 가 `근거없음` 으로 보내는 여섯 개. 여기서 무언가 잡히면 그 결정이 흔들린다.
UNKNOWN_SIX = ("목캔디", "달팽이약", "실리카겔", "매니큐어", "모기향", "계피가루")

#: D-86 이후 모델이 구로 뽑아 오는 것들. 안쪽 어휘가 잡혀야 답할 수 있다.
PHRASES = ("세제 거품", "니코틴 껌 조각", "모기향 태운 재", "감기약")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("words", nargs="*", help="검사할 표면형")
    ap.add_argument("--unknown", action="store_true", help="D-85 여섯 개를 본다")
    ap.add_argument("--phrases", action="store_true", help="구(句) 사례를 본다")
    ap.add_argument("--species", default="dog", help="종 (기본 dog)")
    a = ap.parse_args(argv)

    words = list(a.words)
    if a.unknown:
        words += list(UNKNOWN_SIX)
    if a.phrases:
        words += list(PHRASES)
    if not words:
        words = list(UNKNOWN_SIX) + list(PHRASES)

    from pettriage.compute.vocabulary import mention_in, resolve_substance

    sp = a.species
    print("=" * 92)
    print(f"  어휘표 판독 — 종 {sp!r}")
    print(f"  {'표면형':<16} {'mention_in':<16} {'resolve.name':<24} {'how':<8} 후보")
    print("=" * 92)
    for w in words:
        m = mention_in(w, sp)
        r = resolve_substance(w, sp)
        cand = f"{len(r.candidates)}" + (f" {list(r.candidates)[:2]}" if r.candidates else "")
        print(f"  {w:<16} {str(m):<16} {str(r.name):<24} {str(r.how):<8} {cand}")
    print("=" * 92)
    print("  🔴 `mention_in` 이 값을 내면 그 표면형은 **문장으로 읽혀 부분 매칭됐다**는 뜻이다.")
    print("     D-85 가 거절로 보내야 할 말에서 값이 나오면 그 결정이 무너진다 (D-87 사고).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
