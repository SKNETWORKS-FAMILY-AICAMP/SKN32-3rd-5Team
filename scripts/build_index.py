#!/usr/bin/env python
"""사실 표 → 청크 → 벡터DB 적재.

    python scripts/build_index.py                  # 청크만 만들어 보고 (기본)
    python scripts/build_index.py --store chroma   # 실제 적재 + 검색 점검

설계 근거: docs/02 §3·§11 · docs/06 D-14 · D-20 · D-38 · D-44

**문장화는 코드가 한다.** 여기서 LLM 을 부르지 않는다.
표가 맞으면 문장도 맞는다 — 그래서 검증은 표 단계에서 끝난다 (01e).

적재만 하고 끝내지 않는다. **적재 직후 한국어 질의로 검색을 점검한다** —
"들어갔다"와 "찾아진다"는 다른 문제이고, 후자가 안 되면 그래프도 평가도 돌지 않는다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pettriage import paths  # noqa: E402
from pettriage.config import get_config  # noqa: E402
from pettriage.ingest.facts_io import build_chunks, load_all, summarize  # noqa: E402

#: 적재 직후 돌리는 **양성** 점검 — (보호자가 쓸 법한 말, 물어와야 할 것, 기대 종).
#:
#: 임베딩을 갈거나 문장 템플릿을 고치면 **여기가 먼저 깨진다.**
#: 고양이 질의는 D-39의 병합 검색을 건드린다 —
#: `cat` 자체 자료가 2단계뿐이라 `mammal`·`all` 을 함께 봐야 한다.
#:
#: **처음 7건은 대부분 단어 맞추기였다** (2026-08-01 확장).
#: 질의에 물질명이 그대로 들어 있어(`초콜릿`·`백합`·`아보카도`) 어휘 일치만 확인됐고,
#: 진짜 의미 검색은 PTFE 하나뿐이었는데 **그것만 0.547로 임계값 턱걸이**였다.
#: 우연이 아니다 — 어려운 것을 하나만 넣었기 때문이다. 아래 4종을 보강했다.
#:
#:   ① 영양 · 증상 · 리콜 — `nutrition` 이 274청크로 최대인데 프로브가 0건이었다
#:   ② 물질명이 없는 의미 질의 — 어휘 일치 착시를 걷어낸다
#:   ③ 종 분기 — 개 백합과 고양이 백합은 위험도가 다르다 (D-39)
#:   ④ 검수 회귀 — 고친 값이 되돌아가면 여기서 잡는다
PROBES: tuple[tuple[str, str, str], ...] = (
    # ── 중독 (어휘가 겹치는 쉬운 축) ──────────────────────
    ("강아지가 초콜릿을 먹었어요", "초콜릿", "dog"),
    ("우리 개가 포도를 먹었는데 괜찮을까요", "포도", "dog"),
    ("강아지가 자일리톨 껌을 삼켰어요", "자일리톨", "dog"),
    ("고양이가 백합을 씹었어요", "백합", "cat"),
    ("고양이가 양파 들어간 음식을 먹었어요", "양파", "cat"),
    ("앵무새가 아보카도를 먹었어요", "아보카도", "bird"),
    # ── 물질명이 질의에 없다 (의미 검색 축) ────────────────
    # 실제 보호자는 물질 이름을 모른다. 이 축이 무너지면 코퍼스가 커도 소용없다.
    ("앵무새 앞에서 프라이팬을 태웠어요", "PTFE", "bird"),
    # 물질을 서술만 하는 질의는 여기 없다 — `UNKNOWN_SUBSTANCE_PROBES` 로 옮겼다 (D-49 후속).
    # 기대값이 "세정" 이었을 때 실패로 잡혔는데, 물어온 것은 S-086 "가정용 청소용품(공통)"
    # 으로 **경구 노출 증상까지 맞는 청크**였다. 검색이 아니라 기대값이 틀렸다.
    ("고양이가 화장실 청소하다 흘린 거품을 핥았어요", "청소", "cat"),
    # ── 영양 (274청크. 프로브가 0건이던 자리) ──────────────
    ("강아지 하루 단백질은 얼마나 먹여야 하나요", "단백질", "dog"),
    ("강아지 사료에 칼슘이 얼마나 들어야 하나요", "칼슘", "dog"),
    # ── 리콜 ──────────────────────────────────────────────
    # 증상만 주는 질의는 여기 없다 — `SYMPTOM_PROBES` 로 옮겼다 (D-49).
    ("최근에 회수된 개 사료가 있나요", "리콜", "dog"),
    # ── 종 분기 (D-39) ────────────────────────────────────
    # 개 백합은 위장관 증상뿐이다. 고양이 자료가 넘어오면 **개 보호자에게 과잉 경보**다.
    ("강아지가 백합 잎을 뜯어 먹었어요", "백합", "dog"),
    # ── 검수 회귀 (2026-08-01) ────────────────────────────
    # 남천의 2-2.5 mg/kg 은 **시안화수소 치사량**이지 식물 섭취량이 아니었다.
    # 국화 AFCD 행은 백합 서술이 통째로 복붙돼 있었다.
    ("고양이가 남천 열매를 먹었어요", "남천", "cat"),
    ("고양이가 국화를 씹었어요", "국화", "cat"),
)

#: **찾히면 안 되는** 질의. 1위 점수가 임계값 **미만**이어야 통과다.
#:
#: 양성 프로브만 있으면 "다 잘 찾는다"는 착시가 생긴다.
#: 우리가 실제로 평가받는 것은 **근거가 없을 때 거절하는가**이고 (02 §8.3 · D-46),
#: 임계값을 낮추면 양성은 전부 초록인 채 거절만 조용히 죽는다.
#:
#: `calibrate_threshold.py` 에도 음성이 있지만 그건 따로 돌리는 스크립트라
#: **적재할 때마다 도는 관문이 아니다.**
NEGATIVE_PROBES: tuple[str, ...] = (
    "고양이 캣타워 추천해 주세요",
    "오늘 날씨 어때요",
    "강아지 미용 잘하는 곳 알려주세요",
    "반려동물 보험료가 얼마인가요",
)


#: **증상만 주는 질의.** 통과/실패를 매기지 않고 **모호도를 보고한다** (D-49).
#:
#: 코퍼스는 D-14 로 **물질 단위**라, 각 청크가 "이 물질 → 이런 증상" 이다.
#: 증상만 주면 그 화살표를 거꾸로 타는데 **역방향은 일대일이 아니다.**
#:
#:     '고양이가 토하고 밥을 안 먹고 배를 아파해요'
#:       → 1위 0.616 "고양이에게 토마토는 조건부로 분류된다.
#:                    주요 증상은 … 식욕 부진, 침울, 쇠약 …"
#:
#: 검색은 제대로 일했다 — 증상 목록이 실제로 맞는다.
#: 그런데 고양이 청크 418건 중 **21건이 같은 증상 조합**을 나열하므로,
#: 무엇이 1위가 되든 근거로는 임의다.
#:
#: 여기서 1위를 근거로 답을 만들면 *"토마토 중독일 수 있습니다"* 가 되고,
#: **증상에서 원인을 지목하는 것이 곧 진단이다** (D-11).
#: 그래서 이 질의들의 올바른 처리는 검색이 아니라 ①분류·②되묻기다.
SYMPTOM_PROBES: tuple[tuple[str, str], ...] = (
    ("고양이가 토하고 밥을 안 먹고 배를 아파해요", "cat"),
    ("강아지가 자꾸 침을 흘리고 기운이 없어요", "dog"),
    ("앵무새가 깃털을 부풀리고 바닥에 앉아 있어요", "bird"),
)


def probe_symptom(store, top_k: int) -> None:
    """증상 질의의 **모호도만 보고한다.** 통과/실패를 매기지 않는다.

    여기서 초록·빨강을 매기면 거짓말이 된다 — 올바른 동작은 그래프 노드의 몫이고
    (`intent=symptom` → 되묻기 또는 증상 조합 트리아지),
    이 스크립트는 저장소만 들고 있어 그것을 검증할 수 없다.

    **대신 무엇이 가려져 있는지는 보여준다** — 상위 결과가 서로 다른 물질로
    흩어진다는 사실 자체가, 하나를 골라 답하면 안 된다는 근거다 (04 §8).
    """
    print("\n증상 질의 모호도 — 판정하지 않는다 (D-49)")
    for q, species in SYMPTOM_PROBES:
        where = {"species": [species, "mammal", "all"]}
        hits = store.search(q, top_k=top_k, where=where)
        if not hits:
            print(f"  · {q!r} → 결과 없음")
            continue
        names = [h.chunk.substance or "(무명)" for h in hits]
        spread = hits[0].score - hits[-1].score
        print(f"  · {q!r}")
        print(f"      상위 {len(hits)}: {' · '.join(n[:16] for n in names)}")
        print(f"      서로 다른 물질 {len(set(names))}종 · 점수 폭 {spread:.3f}")
    print("  → 이 질의들은 **물질을 지목하지 않는다.** ①분류가 되묻기로 보내야 한다 (D-11 · D-49)")


#: **물질을 이름이 아니라 서술로만 주는 질의.** 판정하지 않고 보고한다 (D-49 후속).
#:
#: 증상 질의(`SYMPTOM_PROBES`)와 실패 방식이 다르다.
#:
#:   증상 질의   후보 5종이 **0.019 차로 동점** — 하나를 고르면 나머지를 배제한다 = 진단
#:   물질 서술   후보가 **코퍼스에 아예 없다** — 하나를 고르면 근거 없는 추측이다 = 환각
#:
#: 실패 방식은 다르나 **처방은 같다** — 사용자가 모르는 것을 시스템이 추측하지 않는다.
#:
#: `강아지가 차고 바닥에 흘린 달콤한 액체를 핥았어요` 를 넣었더니
#: 1위가 S-080 `앞발·다리 강박 핥기·씹기` 였다 — 질의의 *"핥았어요"* 가
#: 청크의 *"핥기"* 에 표면적으로 걸린 것이다.
#:
#: 그런데 **코퍼스에 `달콤`·`단맛`·`차고` 가 0건이다.** 부동액 청크는
#: *"개에서 에틸렌글리콜(부동액)은 응급 상황이다"* 뿐이고 단맛 서술이 없다.
#: 검색이 이것을 맞힐 방법이 애초에 없다.
#:
#: > **맞혔다면 그게 더 문제다.** 근거가 우리 문서에 없으므로,
#: > 맞혔다는 것은 임베딩의 사전지식이 답을 만들었다는 뜻이다.
#: > 결과가 우연히 옳아도 **환각의 정의 그대로**다.
#:
#: 그래서 이 프로브의 통과 조건은 "찾는다"가 아니다. **못 찾는 것이 정상**이고,
#: 올바른 처리는 ②슬롯의 되묻기다 — 골든셋 `G-014` 가 같은 형태로 `clarify` 다.
#: 정답은 **별칭 묶음**으로 적는다. 한 이름만 적었더니 첫 실행에서
#: `에틸렌글리콜` 을 찾느라 4위의 **`부동액`(S-029)을 놓쳤다** —
#: 보고가 "상위에 없음(정상)" 이라고 말했는데 실제로는 있었다.
#: **검사기가 틀리면 그 초록불이 곧 거짓 근거다** (04 §8).
UNKNOWN_SUBSTANCE_PROBES: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("강아지가 차고 바닥에 흘린 달콤한 액체를 핥았어요", ("에틸렌글리콜", "부동액"), "dog"),
    ("고양이가 베란다에 둔 파란 알갱이를 주워 먹었어요", ("살서제", "쥐약", "살충제"), "cat"),
    ("앵무새가 새장 옆에 뿌린 스프레이를 마셨어요", ("에어로졸", "스프레이", "방향제"), "bird"),
)


def probe_unknown_substance(store, top_k: int) -> None:
    """물질 서술 질의의 **결과만 보고한다.** 통과/실패를 매기지 않는다.

    사람이 아는 정답(`에틸렌글리콜` 등)이 상위에 **없는 것이 정상**이다 —
    코퍼스에 그 서술(단맛·색·형태)이 없기 때문이다.
    있다면 그것은 임베딩 사전지식이 새어 든 것이므로 **오히려 표시해 둘 값**이다.
    """
    print("\n물질 서술 질의 — 판정하지 않는다 (D-49 후속)")
    for q, aliases, species in UNKNOWN_SUBSTANCE_PROBES:
        where = {"species": [species, "mammal", "all"]}
        hits = store.search(q, top_k=top_k, where=where)
        if not hits:
            print(f"  · {q!r} → 결과 없음")
            continue
        names = [h.chunk.substance or "(무명)" for h in hits]
        rank = next(
            (i for i, n in enumerate(names, 1) if any(a in n for a in aliases)),
            None,
        )
        print(f"  · {q!r}")
        print(f"      상위 {len(hits)}: {' · '.join(n[:16] for n in names)}")
        print(f"      서로 다른 물질 {len(set(names))}종")
        if rank:
            print(
                f"      정답 계열({'·'.join(aliases)}) **{rank}위** — "
                "코퍼스 근거로 올라온 것인지 임베딩 사전지식인지 확인할 것"
            )
        else:
            print(
                f"      정답 계열({'·'.join(aliases)}) 상위에 없음 "
                "(정상 — 코퍼스에 서술 근거가 없다)"
            )
    print("  → 이 질의들은 **되묻는다.** 서술로 물질을 특정하면 근거 없는 추측이다 (D-49 후속)")


def probe_negative(store, threshold: float, top_k: int) -> int:
    """음성 점검 — **1위가 임계값을 넘으면 실패다.**

    넘는다는 것은 관계없는 질의가 근거를 얻는다는 뜻이고,
    그러면 파이프라인이 거절해야 할 자리에서 답을 만든다.
    """
    print(f"\n음성 점검 — 1위가 {threshold} **미만**이어야 통과")
    fails = 0
    for q in NEGATIVE_PROBES:
        hits = store.search(q, top_k=top_k)
        if not hits:
            print(f"  ✓ {q!r} → 결과 없음")
            continue
        top = hits[0]
        ok = top.score < threshold
        fails += 0 if ok else 1
        print(f"  {'✓' if ok else '✗'} {q!r}  1위 {top.score:.3f} · {top.chunk.substance}")
        if not ok:
            print("      임계값을 넘었다 — 이 질의가 근거를 얻으면 거절이 안 된다 (D-46)")
    print(f"\n  → 음성 {len(NEGATIVE_PROBES)}건 중 실패 {fails}건")
    return fails


def probe(store, threshold: float, top_k: int) -> int:
    """점검 질의를 돌려 **실패 건수**를 돌려준다.

    통과 기준 두 가지 —
      1. 기대한 것이 상위 `top_k` 안에 있다
      2. 1위 점수가 `score_threshold` 를 넘는다.
         못 넘으면 파이프라인이 그 질의를 **거절로 보낸다** (02 §8.3)
    """
    print(f"\n검색 점검 — top_k={top_k} · score_threshold={threshold}")
    fails = 0
    for q, expect, species in PROBES:
        where = {"species": [species, "mammal", "all"]} if species else None
        hits = store.search(q, top_k=top_k, where=where)
        if not hits:
            print(f"  ✗ {q!r} → 검색 결과 없음")
            fails += 1
            continue
        top = hits[0]
        found = any(expect in h.chunk.substance or expect in h.chunk.text for h in hits)
        ok_score = top.score >= threshold
        ok = found and ok_score
        fails += 0 if ok else 1
        print(f"  {'✓' if ok else '✗'} {q!r}")
        print(f"      1위 {top.score:.3f} · {top.chunk.substance} ({top.source_id})")
        if not found:
            print(f"      기대한 {expect!r} 가 상위 {top_k} 안에 없다")
        if not ok_score:
            print(f"      1위 점수가 임계값 {threshold} 미만 — 이 질의는 거절로 간다")
    print(f"\n  → 점검 {len(PROBES)}건 중 실패 {fails}건")
    if fails:
        print("     임베딩·문장 템플릿·score_threshold 중 하나를 봐야 한다 (configs/default.yaml)")
    return fails


def main() -> int:
    cfg = get_config()
    ap = argparse.ArgumentParser(description="사실 표를 벡터DB에 적재한다")
    ap.add_argument("--facts-dir", type=Path, default=None)
    ap.add_argument(
        "--store",
        choices=["dry-run", "memory", "chroma"],
        default="dry-run",
        help="chroma 는 configs 의 retrieval.persist_dir 에 적재한다 (D-44)",
    )
    ap.add_argument("--out", type=Path, help="청크를 JSONL 로 저장 (검수용)")
    ap.add_argument("--no-probe", action="store_true", help="적재 후 검색 점검을 건너뛴다")
    args = ap.parse_args()

    root = paths.find_root() or Path.cwd()
    facts_dir = args.facts_dir or root / "data" / "facts"
    if not facts_dir.is_dir():
        print(f"✗ 사실 표 폴더가 없다: {facts_dir}")
        return 1

    facts = load_all(facts_dir)
    if not facts:
        print(f"✗ {facts_dir} 에 facts_*.csv 가 없다. 양식: data/facts/사실표_양식.csv")
        return 1

    print(f"사실 {len(facts)}건\n")
    dist = summarize(facts)
    for k, v in dist.items():
        print(f"  {k:<16} {v}")

    # 종별 쏠림은 골든셋 설계를 무너뜨린다 (04 §2.3)
    if dist["species"].get("bird", 0) == 0:
        print("\n  ⚠ 조류 0건 — 04 §2.3 종별 최소 건수를 만족할 수 없다")

    chunks = build_chunks(facts)
    print(f"\n청크 {len(chunks)}건 생성 (물질 단위 · D-14)")

    empty = [c for c in chunks if len(c.text.strip()) < 20]
    if empty:
        print(f"  ⚠ 문장이 지나치게 짧은 청크 {len(empty)}건 — 사실 표가 비어 있을 수 있다")
        for c in empty[:3]:
            print(f"      {c.chunk_id}: {c.text!r}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
        print(f"  → {args.out}")

    if args.store == "dry-run":
        print("\n예시 문장 3건")
        for c in chunks[:3]:
            print(f"  [{c.species}/{c.doc_type}] {c.text}")
        print("\n적재하려면 --store chroma")
        return 0

    from pettriage.retrieval import ChromaStore, HashEmbedder, InMemoryStore, get_embedder

    r = cfg.retrieval
    if args.store == "memory":
        # 모델 없이 배선만 확인한다. **검색 품질은 이 경로로 판단할 수 없다.**
        store = InMemoryStore(embedder=HashEmbedder())
        print("\n⚠ HashEmbedder — 배선 확인용이다. 검색 품질 판단에 쓰지 않는다")
    else:
        print(f"\n임베딩 모델 로딩: {r.embedding_model}")
        print("  처음이면 모델을 내려받는다 (bge-m3 약 2.2GB). 몇 분 걸릴 수 있다.")
        store = ChromaStore(
            embedder=get_embedder(r.embedding_model),
            persist_dir=str(root / r.persist_dir),
            collection=r.collection,
        )

    t0 = time.time()
    n = store.add(chunks)
    print(f"적재 {n}건 → {store.name} (총 {store.count()}건) · {time.time() - t0:.1f}s")
    if args.store == "chroma":
        print(f"  위치: {root / r.persist_dir}  ·  컬렉션: {r.collection}")

    if args.no_probe:
        return 0

    # 양성·음성을 **둘 다** 돌린다. 하나만 보면 임계값을 잘못 잡아도 초록이 나온다.
    fails = probe(store, r.score_threshold, r.top_k)
    fails += probe_negative(store, r.score_threshold, r.top_k)
    probe_symptom(store, r.top_k)  # 판정하지 않는 보고 (D-49)
    probe_unknown_substance(store, r.top_k)  # 판정하지 않는 보고 (D-49 후속)
    print(f"\n{'✓ 전체 통과' if not fails else f'✗ 총 실패 {fails}건'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
