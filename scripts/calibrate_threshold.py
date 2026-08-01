#!/usr/bin/env python
"""유사도 임계값 보정 — `retrieval.score_threshold` 를 근거로 정한다.

    python scripts/calibrate_threshold.py

설계 근거: docs/02 §8.3·§9 · docs/06 D-44 · docs/04 §4

**임계값 미만이면 파이프라인이 그 질의를 거절로 보낸다** (02 §8.3).
너무 낮으면 무관한 근거로 답을 만들고, 너무 높으면 있는 근거를 두고 거절한다.
둘 다 지표를 망치므로 **추측하지 않고 측정해서 정한다.**

방법은 단순하다 — **답이 있어야 하는 질의**와 **없어야 하는 질의**를 각각 던져
점수 분포가 갈리는지 본다. 갈리면 그 사이가 임계값이고, 겹치면 임계값만으로는
거절을 만들 수 없다는 뜻이다.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pettriage import paths  # noqa: E402
from pettriage.config import get_config  # noqa: E402

#: 코퍼스에 **근거가 있는** 질의. 여기서 낮은 점수가 나오면 임계값을 올릴 수 없다.
POSITIVE: tuple[tuple[str, str], ...] = (
    ("강아지가 초콜릿을 먹었어요", "dog"),
    ("우리 개가 포도를 먹었는데 괜찮을까요", "dog"),
    ("강아지가 자일리톨 껌을 삼켰어요", "dog"),
    ("개가 마카다미아를 주워 먹었어요", "dog"),
    ("강아지가 양파 들어간 국물을 핥았어요", "dog"),
    ("고양이가 백합을 씹었어요", "cat"),
    ("고양이가 양파 들어간 음식을 먹었어요", "cat"),
    ("고양이가 숨을 헐떡여요", "cat"),
    ("앵무새가 아보카도를 먹었어요", "bird"),
    ("앵무새 앞에서 프라이팬을 태웠어요", "bird"),
    ("앵무새가 초콜릿을 쪼아 먹었어요", "bird"),
    ("강아지가 갑자기 발작을 해요", "dog"),
)

#: 코퍼스에 **근거가 없는** 질의. 이것들이 높은 점수를 받으면 임계값이 방어를 못 한다.
#:
#: 세 종류를 섞었다 —
#:   ① 도메인 밖   ② 도메인 안이지만 코퍼스에 없는 주제   ③ 물질은 있으나 물을 수 없는 것
NEGATIVE: tuple[tuple[str, str, str], ...] = (
    ("고양이 이름 좀 지어주세요", "cat", "도메인 밖"),
    ("강아지 배변 훈련은 어떻게 하나요", "dog", "도메인 밖"),
    ("앵무새 말 가르치는 방법 알려줘", "bird", "도메인 밖"),
    ("반려동물 보험은 어디가 좋나요", "dog", "도메인 밖"),
    ("강아지 예방접종 일정이 어떻게 되나요", "dog", "코퍼스 밖"),
    ("고양이 중성화 수술 비용이 궁금해요", "cat", "코퍼스 밖"),
    ("앵무새 발톱은 얼마나 자주 깎아야 하나요", "bird", "코퍼스 밖"),
    ("강아지가 산책 중에 다른 개를 보면 짖어요", "dog", "코퍼스 밖"),
    ("앵무새 체중 100g 기준 초콜릿 몇 g부터 위험한가요", "bird", "조류 정량 0건"),
    ("우리 강아지 사료 브랜드 추천해주세요", "dog", "도메인 밖"),
)


def _scores(store, queries, top_k: int) -> list[tuple[str, float, str, str]]:
    out = []
    for q, species, *rest in queries:
        where = {"species": [species, "mammal", "all"]} if species else None
        hits = store.search(q, top_k=top_k, where=where)
        top = hits[0] if hits else None
        out.append(
            (
                q,
                top.score if top else 0.0,
                top.chunk.substance if top else "(결과 없음)",
                rest[0] if rest else "",
            )
        )
    return out


def main() -> int:
    cfg = get_config()
    r = cfg.retrieval
    root = paths.find_root() or Path.cwd()

    from pettriage.retrieval import ChromaStore, get_embedder

    store = ChromaStore(
        embedder=get_embedder(r.embedding_model),
        persist_dir=str(root / r.persist_dir),
        collection=r.collection,
    )
    n = store.count()
    if n == 0:
        print("✗ 벡터DB가 비어 있다. 먼저: python scripts/build_index.py --store chroma")
        return 1
    print(f"코퍼스 {n}건 · 임베딩 {r.embedding_model} · 현재 임계값 {r.score_threshold}\n")

    pos = _scores(store, POSITIVE, r.top_k)
    neg = _scores(store, NEGATIVE, r.top_k)

    print("── 근거가 있어야 하는 질의 ──────────────────────────")
    for q, s, sub, _ in sorted(pos, key=lambda x: x[1]):
        print(f"  {s:.3f}  {q[:36]:38} → {sub[:30]}")
    print("\n── 근거가 없어야 하는 질의 ──────────────────────────")
    for q, s, sub, why in sorted(neg, key=lambda x: -x[1]):
        print(f"  {s:.3f}  {q[:36]:38} → {sub[:24]:26} [{why}]")

    p_min, p_max = min(s for _, s, _, _ in pos), max(s for _, s, _, _ in pos)
    n_min, n_max = min(s for _, s, _, _ in neg), max(s for _, s, _, _ in neg)

    print("\n── 분포 ────────────────────────────────────────────")
    print(f"  근거 있음 : {p_min:.3f} ~ {p_max:.3f}")
    print(f"  근거 없음 : {n_min:.3f} ~ {n_max:.3f}")

    print("\n── 판단 ────────────────────────────────────────────")
    if n_max < p_min:
        gap = p_min - n_max
        rec = round(n_max + gap / 2, 2)
        print(f"  ✓ 두 분포가 갈린다 (간격 {gap:.3f}).")
        print(f"    권장 임계값 ≈ {rec}  — 둘 사이 한가운데")
        print("    안전 쪽으로 기울이려면 조금 **낮게** 잡는다.")
        print("    임계값을 높이면 근거가 있는데도 거절해 과소평가로 이어질 수 있다 (D-13).")
    else:
        over = [(q, s, why) for q, s, _, why in neg if s >= p_min]
        print(f"  ⚠ 두 분포가 겹친다. 근거 없는 질의 {len(over)}건이 근거 있는 최저점 이상이다.")
        for q, s, why in sorted(over, key=lambda x: -x[1])[:5]:
            print(f"      {s:.3f}  {q[:40]} [{why}]")
        print("\n    **임계값만으로는 거절을 만들 수 없다.** 아래를 함께 써야 한다 —")
        print("      · ① 의도 분류에서 도메인 밖을 먼저 걸러낸다 (02 §6)")
        print("      · ④ 근거 검증에서 문장별로 근거 유무를 판정한다 (D-11)")
        print("    임계값은 '무관한 근거로 답을 만들지 않는' 최소 방어선으로만 둔다.")

    cur = r.score_threshold
    fn = [q for q, s, _, _ in pos if s < cur]
    fp = [q for q, s, _, _ in neg if s >= cur]
    print(f"\n  현재 임계값 {cur} 로는 —")
    print(f"    근거 있는데 거절될 질의 {len(fn)}건" + (f": {fn[0][:34]}…" if fn else ""))
    print(f"    근거 없는데 통과할 질의 {len(fp)}건" + (f": {fp[0][:34]}…" if fp else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
