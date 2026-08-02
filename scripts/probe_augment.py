#!/usr/bin/env python
"""검색어 보강이 **실제 임베딩에서** 점수를 올리나 내리나 (D-62 · 2026-08-02 흡수).

    python scripts/probe_augment.py            # 적재된 Chroma 를 읽는다

왜 재야 하나
-----------
`retrieve` 가 질의에 정규화된 물질명을 덧붙이게 바꿨다 (`대파` → `… 알리움류`).
별칭 표가 검색에 닿게 하려는 것인데, **덧붙이면 벡터가 움직인다.**

    올라가면   의도한 대로다
    내려가면   D-59 ② 가 든 세 번째 실패다 — *"확장어가 검색을 엉뚱한 쪽으로 끌고 간다"*

`HashEmbedder` 로는 답을 알 수 없다. **실제 임베딩으로만 재진다.**
그리고 임계값 `0.50` 을 넘는지도 여기서 함께 본다 (D-46).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pettriage import paths  # noqa: E402
from pettriage.config import get_config  # noqa: E402
from pettriage.graph.nodes import build_filter, classify_intent, extract_slots  # noqa: E402
from pettriage.graph.nodes.retrieve import _augment  # noqa: E402
from pettriage.graph.state import initial_state  # noqa: E402

QUERIES: tuple[str, ...] = (
    "5kg 고양이가 대파를 40g쯤 뜯어 먹었어요",
    "저희 앵무새가 프라이팬에서 나는 연기를 마신 것 같아요",
    "강아지가 무설탕껌을 삼켰어요",
    "고양이한테 우유를 매일 조금씩 줘도 되나요",
    "강아지가 차고에서 부동액을 핥은 것 같아요",
    "고양이가 백합 꽃잎을 씹어먹었어요",
    "6kg 강아지가 제빵용 무가당 초콜릿 20g을 먹었어요",
    "앵무새가 조리 중인 냄비 위에 앉았다가 발바닥이 벗겨졌어요",
)


def main() -> int:
    from pettriage.retrieval import ChromaStore, get_embedder

    root = paths.find_root() or Path.cwd()
    r = get_config().retrieval
    print(f"임베딩 {r.embedding_model} · top_k {r.top_k} · 임계값 {r.score_threshold}")
    store = ChromaStore(
        embedder=get_embedder(r.embedding_model),
        persist_dir=str(root / r.persist_dir),
        collection=r.collection,
    )
    print(f"적재 {store.count()}건\n")

    worse = 0
    for q in QUERIES:
        st = initial_state(q, "probe")
        st.update(classify_intent(st))
        st.update(extract_slots(st))
        st.update(build_filter(st))
        aug = _augment(st)
        where = st.get("where") or None

        a = store.search(q, top_k=r.top_k, where=where)
        b = store.search(aug, top_k=r.top_k, where=where)
        sa = a[0].score if a else 0.0
        sb = b[0].score if b else 0.0
        d = sb - sa
        if d < -0.005:
            worse += 1
        mark = "↑" if d > 0.005 else ("↓" if d < -0.005 else "=")
        gate = "통과" if sb >= r.score_threshold else "**임계 미만**"
        print(f"  {q[:34]:34}")
        print(f"      원문 {sa:.3f} → 보강 {sb:.3f}  {mark}{d:+.3f}  {gate}")
        if aug != q:
            print(f"      보강어: {aug[len(q):].strip()}")
        if b:
            print(f"      1위: {b[0].chunk.substance[:44]}")

    print(f"\n→ 보강이 점수를 낮춘 질의 {worse}/{len(QUERIES)}건")
    if worse:
        print("  낮아졌다면 D-59 ② 의 세 번째 실패다 — 보강어가 원문 신호를 밀어낸다.")
        print("  그때는 보강을 '검색어'가 아니라 'where 필터'로 옮기는 것을 본다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
