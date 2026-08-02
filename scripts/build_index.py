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
#: **D-46 이 정한 임계값 하한.** 이 아래로 내리면 근거가 없다.
#:
#: 임계값은 두 방향으로 틀릴 수 있고, 각각 다른 장치가 잡는다.
#:
#:   올리면 → 근거 있는 질의가 잘린다 (과소평가, D-13). **양성 프로브**가 잡는다
#:   내리면 → 0.2대 문서로 답하는 경로가 열린다. **이 상수**가 잡는다
#:
#: D-46 이 (c)"임계값을 없앤다"를 버린 이유가 후자다 —
#: *"임계값이 없으면 검색이 항상 무언가를 돌려준다."* 최소선은 남긴다.
THRESHOLD_FLOOR: float = 0.50


def probe_threshold(threshold: float) -> int:
    """임계값이 D-46 의 하한 아래로 내려갔는지 본다. **설정값은 인덱서가 판정할 수 있다** (D-58)."""
    if threshold >= THRESHOLD_FLOOR:
        return 0
    print(f"\n✗ score_threshold={threshold} 가 D-46 하한 {THRESHOLD_FLOOR} 미만이다.")
    print("  내리려면 `scripts/calibrate_threshold.py` 로 재측정하고 D-46 을 갱신할 것.")
    print("  근거 없이 내리면 0.2대 문서로 답하는 경로가 열린다 (configs/default.yaml).")
    return 1


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


def probe_negative(store, top_k: int, positive_scores: list[float]) -> None:
    """음성 점검 — **보고만 한다. 판정하지 않는다** (D-46 · D-58).

    ⚠️ 2026-08-02 에 판정을 걷어냈다. 예전에는 이랬다.

        print(f"음성 점검 — 1위가 {threshold} **미만**이어야 통과")
        ok = top.score < threshold
        fails += 0 if ok else 1
        # 실패 메시지: "임계값을 넘었다 — 이 질의가 근거를 얻으면 거절이 안 된다 (D-46)"

    **D-46 을 인용하면서 D-46 이 폐기한 기준을 강제하고 있었다.** D-46 은 실측으로
    근거 있음(0.547~0.733)과 없음(0.494~0.659)이 **겹친다**는 것을 확인하고,
    `0.50` 을 고르면서 *"근거 없는 것 중 최저 1건만 차단 · 방어 효과는 거의 없다 ·
    **그 사실을 숨기지 않는다**"* 라고 적었다. 10건 중 1건이 예상값인데
    이 검사는 4건 중 4건을 요구했다 — **상시 빨간불**이고, 상시 빨간불은 아무도 안 본다.

    거절은 임계값이 아니라 ①`classify_intent`(범위밖)와 ④`verify_grounding`(근거없음)이
    만든다. 검색 단계에서 판정할 수 있는 것이 아니다 (D-58).

    그래서 여기서 재는 것은 **겹침 구간**이다. D-46 의 근거가 되는 숫자이고,
    인덱스를 다시 만들 때마다 재측정되어야 한다. 겹침이 사라지면 그것이 뉴스다.
    """
    print("\n음성 점검 — **보고만 한다.** 임계값은 거절을 만들지 못한다 (D-46)")
    negative_scores: list[float] = []
    for q in NEGATIVE_PROBES:
        hits = store.search(q, top_k=top_k)
        if not hits:
            print(f"  · {q!r} → 결과 없음")
            continue
        top = hits[0]
        negative_scores.append(top.score)
        print(f"  · {q!r}  1위 {top.score:.3f} · {top.chunk.substance}")

    if not (positive_scores and negative_scores):
        print("\n  ▸ 겹침을 계산할 표본이 부족하다")
        return
    pos_lo, pos_hi = min(positive_scores), max(positive_scores)
    neg_hi = max(negative_scores)
    print(f"\n  ▸ 양성 {pos_lo:.3f}~{pos_hi:.3f}  /  음성 최고 {neg_hi:.3f}")
    if neg_hi >= pos_lo:
        print("    두 분포가 **겹친다** — D-46 재확인. 어떤 임계값도 이 둘을 가르지 못한다.")
        print("    거절은 ①분류(범위밖)와 ④검증(근거없음)이 만든다. 이 값으로 만들지 않는다.")
    else:
        print("    겹치지 않는다 — **D-46 의 전제가 바뀌었다.** 재측정하고 결정을 갱신할 것.")


def probe(store, threshold: float, top_k: int) -> tuple[int, list[float]]:
    """점검 질의를 돌려 **실패 건수와 1위 점수 목록**을 돌려준다.

    통과 기준 두 가지 — **검색이 통제하는 것만 본다** (D-58).

      1. 기대한 것이 상위 `top_k` 안에 있다 (`recall@k`)
      2. 1위 점수가 `score_threshold` 를 넘는다. 못 넘으면 임계 필터가 전부 걷어내
         **결과 0건**이 되고, 0건은 거절 신호다 (D-46 — 0건**만**이 거절이다)

    ⚠️ **순위로 판정하지 않는다.** 기대값이 1위인지 3위인지는 검색의 계약이 아니다 —
    `retrieve` 는 `{"hits": [...]}` 로 **후보 집합 전체**를 넘기고, 그 뒤
    `compress_context`·`generate_draft` 가 상위 k 를 **전부** 읽는다.
    등급은 `triage` 가 `apply_gate` 로 정한다. 순위를 여기서 판정하면
    *"1위가 곧 답"* 이라는, 파이프라인에 존재하지 않는 모델을 강제하게 된다 (D-58).

    (2026-08-02 검토 중 *"1위 등급 < 기대 등급이면 실패"* 를 넣으려다 취소했다.
    등급 판정은 그래프와 평가 하네스의 일이고, 골든셋 `G-013` 이 이미 그 자리에 있다.)

    아래 `기대` 줄은 **진단용**이다 — 어떤 청크가 후보에 들어왔는지 사람이 본다.
    """
    print(f"\n검색 점검 — top_k={top_k} · score_threshold={threshold}")
    print("  판정은 recall@k 와 1위 점수 둘뿐이다. 순위·등급은 판정하지 않는다 (D-58)")
    fails = 0
    top_scores: list[float] = []
    for q, expect, species in PROBES:
        where = {"species": [species, "mammal", "all"]} if species else None
        hits = store.search(q, top_k=top_k, where=where)
        if not hits:
            print(f"  ✗ {q!r} → 검색 결과 없음")
            fails += 1
            continue
        top = hits[0]
        top_scores.append(top.score)
        matched = [
            (i, h)
            for i, h in enumerate(hits, 1)
            if expect in h.chunk.substance or expect in h.chunk.text
        ]
        ok_score = top.score >= threshold
        ok = bool(matched) and ok_score
        fails += 0 if ok else 1
        print(f"  {'✓' if ok else '✗'} {q!r}")
        print(f"      1위 {top.score:.3f} · {top.chunk.substance} ({top.source_id})")
        _print_expected(expect, matched, top_k)
        if not matched:
            print(f"      기대한 {expect!r} 가 상위 {top_k} 안에 없다")
        if not ok_score:
            print(f"      1위 점수가 임계값 {threshold} 미만 — 임계 필터가 0건을 만든다")
    print(f"\n  → 점검 {len(PROBES)}건 중 실패 {fails}건")
    if fails:
        print("     임베딩·문장 템플릿·score_threshold 중 하나를 봐야 한다 (configs/default.yaml)")
    return fails, top_scores


def _print_expected(expect: str, matched: list, top_k: int) -> None:
    """기대 계열이 후보 안에 **무엇으로** 들어왔는지 보여준다. **판정하지 않는다.**

    문자열 하나(`"PTFE"`)에 걸리는 청크가 여러 개이고 **내용이 천차만별**이라 필요하다.
    실측 (2026-08-02) — `PTFE` 계열 6건 중 3건은 등급도 증상도 없다.

        F-071-005  응급   증상 4종   PTFE(폴리테트라플루오로에틸렌) 과열 흄
        F-093-008  없음   증상 0종   PTFE(테플론) 흄        ← 문장이 사실상 비어 있다

    `recall@k` 는 **아무거나 하나**만 들어와도 통과한다. 그것이 위협을 말하는 청크인지는
    사람이 봐야 한다 — *"근거가 충분한가"* 는 ④`verify_grounding` 의 판단이지
    인덱서의 판단이 아니다 (D-58). 그래서 **보여주기만** 한다.

    등급 없는 청크가 있는 것은 결함이 아니다. 원문이 등급을 주지 않았으면 만들지 않는다
    (D-38 · `Fact.triage_ko`).
    """
    if not matched:
        return
    print(f"      기대 {expect!r} 후보 안 {len(matched)}건  (② 슬롯 확장 **전** 기준)")
    for rank, h in matched:
        c = h.chunk
        print(f"        {rank}/{top_k}위 {h.score:.3f} · {c.source_id} · {c.substance[:40]}")
    last = max(r for r, _ in matched)
    if last >= top_k:
        print(f"        ⚠ 마지막이 {last}/{top_k}위 — 자료가 늘거나 중복이 접히면 후보에서 빠진다")


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

    # 판정하는 것 둘 — recall@k(양성)와 임계값 하한. **검색·설정이 통제하는 것뿐이다** (D-58).
    fails = probe_threshold(r.score_threshold)
    probe_fails, positive_scores = probe(store, r.score_threshold, r.top_k)
    fails += probe_fails
    # 아래는 전부 **보고**다. 판정하지 않는다.
    probe_negative(store, r.top_k, positive_scores)  # 겹침 구간 재측정 (D-46)
    probe_symptom(store, r.top_k)  # 증상 질의 모호도 (D-49)
    probe_unknown_substance(store, r.top_k)  # 물질 서술 질의 (D-49 후속)
    print(f"\n{'✓ 전체 통과' if not fails else f'✗ 총 실패 {fails}건'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
