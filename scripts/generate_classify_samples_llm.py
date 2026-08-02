#!/usr/bin/env python
"""① 분류 태스크 — distillation 본작업 (LLM 질의 생성 + LLM 라벨링).

설계 근거: docs/03_모델-멀티태스크학습.md §4 · docs/03a_파인튜닝-구현기획.md §3

    `generate_classify_samples.py`(고정 템플릿)는 조합 공간이 작아 32건
    파일럿까진 됐지만 목표(2,000~5,000건)엔 못 미친다 (증상/영양/일반은
    종 × 템플릿 조합이 16가지뿐). 이 스크립트는 **질문 생성 자체를 대형
    LLM에게 맡긴다** — 03 §4의 "보호자 질의 생성"을 문자 그대로 구현한 것.

    두 단계 다 LLM을 쓰지만 **역할이 다르다**:
      ① 생성 — 종·물질/증상 시드를 던져주고 다양한 문장을 받는다 (창작)
      ② 라벨링 — `graph.nodes.classify._llm_classify`, 즉 **실제 프로덕션
         분류 프롬프트**로 라벨을 매긴다 (판정). 생성 의도(intent_hint)와
         라벨링 결과가 다르면 실제 라벨을 신뢰하고 `disagree`로 표시해
         사람이 검수하게 한다 — 생성기가 의도한 카테고리를 만드는 데
         실패했을 수도 있고, 실제로 애매한 경계 케이스일 수도 있다.

    골든셋 질문·이미 만든 질문과 겹치면 버린다 (D-29 누수 방지).

    python scripts/generate_classify_samples_llm.py --target 750 \
        --out data/train/classify_batch.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_SPECIES_KO = {"dog": "강아지", "cat": "고양이", "bird": "앵무새"}

#: 생성 프롬프트에 "이런 걸 다뤄라" 시드로만 준다 — 강제 목록이 아니다.
_SEEDS: dict[str, str] = {
    "intoxication": (
        "초콜릿·포도·건포도·양파·마늘·자일리톨·아보카도·커피·부동액·백합·알로에·국화·"
        "사람 약(아스피린·감기약)·세제·살충제·목캔디·마카다미아·튤립 구근 등 다양한 물질"
    ),
    "symptom": "구토·설사·기침·발작·떨림·무기력·식욕부진·배 부풀어오름·절뚝거림·피부 발진 등 증상",
    "nutrition": "사료 급여량·간식·칼슘/단백질 등 영양소·체중 관리·비만·기호성·이유식 전환 등",
    "general": "이름 짓기·미용실·보험·훈련 용품·산책 코스·입양 절차 등 건강과 무관한 화제",
}

_INTENT_KO = {
    "intoxication": "중독·오섭취(위험할 수 있는 물질을 먹었거나 접촉한 상황)",
    "symptom": "증상 호소(이미 아파 보이는 상태를 설명)",
    "nutrition": "영양·급여(먹여도 되는지, 얼마나 먹여야 하는지)",
    "general": "건강·응급과 무관한 일반 화제",
}

_GEN_SYSTEM = (
    "너는 반려동물 헬스케어 서비스의 학습 데이터를 만드는 도구다.\n"
    "실제 보호자가 챗봇에 입력할 법한, 자연스러운 한국어 질문을 다양하게 만든다.\n"
    "같은 문장 구조를 반복하지 않는다. 종(개/고양이/앵무새)을 섞고, 일부는 종을 "
    "밝히지 않는 질문(예: '반려동물이 ~')도 만든다.\n"
    '출력은 JSON 배열만. 각 원소는 {"question": str, "species": "dog"|"cat"|"bird"|null} '
    "형식이다. 설명·코드블록 없이 배열만 출력한다."
)


def _gen_prompt(intent: str, n: int, avoid: list[str]) -> str:
    avoid_block = ""
    if avoid:
        sample = avoid[-15:]  # 프롬프트 길이 제한 — 최근 것만 보여준다
        avoid_block = "\n\n이미 만든 문장(반복하지 말 것):\n" + "\n".join(f"- {q}" for q in sample)
    return (
        f"카테고리: {_INTENT_KO[intent]}\n"
        f"참고 소재(전부 쓸 필요 없음, 다양하게 섞을 것): {_SEEDS[intent]}\n"
        f"이 카테고리에 해당하는 질문을 {n}개 만들어라."
        f"{avoid_block}"
    )


def _parse_json_array(raw: str) -> list[dict]:
    """LLM 출력에서 JSON 배열만 뽑는다. 코드블록·잡담이 섞여도 견딘다."""
    text = raw.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if isinstance(item, dict) and item.get("question"):
            species = item.get("species")
            out.append(
                {
                    "question": str(item["question"]).strip(),
                    "species": species if species in _SPECIES_KO else None,
                }
            )
    return out


def generate_for_intent(intent: str, target: int, batch_size: int, seen: set[str]) -> list[dict]:
    from pettriage.models.serving.factory import get_client

    client = get_client()
    if client is None:
        raise RuntimeError("LLM 클라이언트가 없다 — OPENAI_API_KEY 확인할 것")

    out: list[dict] = []
    recent: list[str] = []
    max_rounds = (target // batch_size + 1) * 3  # 중복 재시도 감안한 여유
    rounds = 0
    while len(out) < target and rounds < max_rounds:
        rounds += 1
        want = min(batch_size, target - len(out))
        prompt = _gen_prompt(intent, want, recent)
        raw = client.run_raw(_GEN_SYSTEM, prompt, max_tokens=1600)
        items = _parse_json_array(raw)
        added = 0
        for item in items:
            q = item["question"]
            if not q or q in seen:
                continue
            seen.add(q)
            out.append({"question": q, "species": item["species"], "intent_hint": intent})
            recent.append(q)
            added += 1
        recent = recent[-15:]
        print(f"  [{intent}] round {rounds}: +{added} (누적 {len(out)}/{target})")
        if added == 0 and not items:
            print(f"  ⚠ [{intent}] 생성 응답 파싱 실패 — 원문 앞부분: {raw[:120]!r}")
    if len(out) < target:
        print(f"⚠ {intent}: {target}건 목표, {len(out)}건만 생성됨 (재시도 한도 도달)")
    return out


def _load_goldenset_questions() -> set[str]:
    out: set[str] = set()
    for path in sorted((ROOT / "eval" / "goldenset").glob("golden_*.csv")):
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                q = (row.get("question") or "").strip()
                if q:
                    out.add(q)
    return out


def label_with_teacher(question: str) -> tuple[str, str]:
    from pettriage.graph.nodes.classify import ALLOWED_INTENTS, _llm_classify
    from pettriage.models.serving.factory import client_name

    raw = _llm_classify(question)
    label = raw if raw in ALLOWED_INTENTS else "unknown"
    return label, client_name()


def main() -> int:
    ap = argparse.ArgumentParser(description="① 분류 태스크 distillation — LLM 생성판")
    ap.add_argument("--target", type=int, default=750, help="의도(4종)당 목표 건수")
    ap.add_argument("--batch-size", type=int, default=20, help="생성 1회 호출당 요청 개수")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "classify_batch.jsonl")
    args = ap.parse_args()

    golden_questions = _load_goldenset_questions()
    seen: set[str] = set(golden_questions)  # 골든셋도 seen에 넣어 애초에 안 만들게 한다

    all_candidates: list[dict] = []
    for intent in ("intoxication", "symptom", "nutrition", "general"):
        print(f"=== {intent} 생성 시작 (목표 {args.target}) ===")
        got = generate_for_intent(intent, args.target, args.batch_size, seen)
        all_candidates.extend(got)

    print(f"\n생성 완료: 총 {len(all_candidates)}건 → 라벨링 시작")

    rows_out = []
    mismatches = 0
    for i, c in enumerate(all_candidates, start=1):
        label, teacher = label_with_teacher(c["question"])
        agree = label == c["intent_hint"]
        if not agree:
            mismatches += 1
        rows_out.append(
            {
                "question": c["question"],
                "species": c["species"],
                "intent_hint": c["intent_hint"],
                "teacher_label": label,
                "agree": agree,
                "teacher": teacher,
            }
        )
        if i % 100 == 0:
            print(f"  라벨링 {i}/{len(all_candidates)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n생성+라벨링 {len(rows_out)}건 → {args.out}")
    print(f"생성 의도 vs 교사(LLM) 라벨 불일치: {mismatches}/{len(rows_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
