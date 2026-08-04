#!/usr/bin/env python
"""② 슬롯 추출 — distillation (LLM 질의+정답 동시 생성 → 프로덕션 프롬프트로 교차검증).

설계 근거: docs/03_모델-멀티태스크학습.md §4 · docs/06 D-38

    ① 분류와 다른 점 — ②의 "정답"은 카테고리가 아니라 **문장 안에 실제로
    있는 값**이다. 그래서 생성 단계에서 "이런 슬롯을 담은 문장을 만들어라"고
    시키면서 **정답(슬롯 값)도 같이 받는다** — 만든 사람이 정답을 제일 잘 안다.

    그 정답을 무조건 믿지 않는다. 실제 프로덕션 슬롯 프롬프트
    (`graph.nodes.slots._llm_slots`)로 **같은 문장을 다시 뽑아** 비교한다.
    둘이 다르면 `agree=false` — 문장이 애매했거나 교사가 틀렸다는 뜻이고,
    사람이 봐야 한다 (03 §4 사람 검수).

    **결측 케이스를 의도적으로 섞는다** (03 §4 표) — `_PATTERNS` 가 그 축이다.
    종·체중·섭취량·물질이 전부 있는 문장만 만들면, 모델이 "뭐든 다 있다"고
    학습해 실제 대화의 결측 상황(가장 흔한 경우)에서 무너진다.

    python scripts/generate_slot_samples_llm.py --target 600 \
        --out data/train/slot_batch.jsonl
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

_SPECIES = ("dog", "cat", "bird")

#: 슬롯 결측 패턴 — 03 §4 "결측 케이스를 의도적으로 포함"의 구현.
#: 각 패턴은 (설명, 무엇을 반드시 넣고 무엇을 반드시 비울지)를 프롬프트로 번역한다.
_PATTERNS: dict[str, str] = {
    "full_dose": (
        "종·체중(kg)·섭취량(g)·구체적인 물질명이 **전부 문장에 들어간다.** "
        "예: '5kg 말티즈가 초콜릿 20g을 먹었어요' 같은 형태. 물질은 실제 위험할 수 있는 "
        "것(초콜릿·포도·양파·마늘·자일리톨·아보카도·백합·부동액 등)으로 다양하게."
    ),
    "species_substance_no_amount": (
        "종과 물질명은 있지만 **체중·섭취량 숫자는 절대 언급하지 않는다.** "
        "예: '강아지가 초콜릿을 먹었어요, 위험한가요?' — 얼마나 먹었는지는 안 나온다."
    ),
    "substance_no_species": (
        "물질명은 있지만 **종(개/고양이/앵무새)을 밝히지 않는다.** "
        "'저희 아이가', '반려동물이' 처럼 종 불명 표현을 쓴다. "
        "예: '반려동물이 포도를 먹었는데 괜찮을까요'"
    ),
    "species_no_substance": (
        "종은 있지만 **무엇을 먹었는지는 모른다.** "
        "예: '강아지가 산책하다 뭔가를 주워 먹었는데 뭔지 모르겠어요'"
    ),
    "symptom_species_only": (
        "종과 증상만 있고 물질·체중·섭취량은 전혀 없다. " "예: '고양이가 계속 구토를 해요'"
    ),
    "vague_no_species": (
        "종도 밝히지 않고 막연한 상태만 설명한다. " "예: '반려동물이 기운이 없어 보여요'"
    ),
    "weight_only_nutrition": (
        "종과 체중(kg)은 있지만 물질·섭취량은 없는 **영양/급여량 질문**. "
        "예: '4kg 고양이 사료를 하루에 얼마나 줘야 하나요'"
    ),
    # 2026-08-03 D-89(한빈) — SLOT 출력 스키마에 elapsed_hours가 명시됐는데
    # (D-86) 이 축의 학습 데이터가 단 한 건도 없었다(이서은 팀원 발견).
    # 종·물질은 있고 **경과 시간이 자연어로 섞여 들어가는** 문장을 추가한다.
    "elapsed_hours_mentioned": (
        "종과 물질명이 있고, **먹은 뒤 지난 시간을 자연스럽게 언급한다.** "
        "'3시간 전에', '어제 저녁에', '한 30분 됐어요', '방금' 처럼 다양한 "
        "표현으로 시간을 담되, 문장에는 **숫자로 환산 가능한 단서**가 있어야 한다 "
        "(예: '어제 저녁 8시쯤' 은 지금이 언제인지 몰라 숫자화가 안 되니 피하고, "
        "'12시간 전' · '한 시간쯤 됐어요' 처럼 경과 시간 자체를 말하게 한다). "
        "체중·섭취량은 언급해도 되고 안 해도 된다. "
        "예: '강아지가 3시간 전에 초콜릿을 먹었어요', '고양이가 한 30분 전에 백합을 씹었어요'"
    ),
}

_GEN_SYSTEM = (
    "너는 반려동물 헬스케어 서비스의 학습 데이터를 만드는 도구다.\n"
    "지시된 패턴에 맞는, 실제 보호자가 쓸 법한 자연스러운 한국어 문장을 여러 개 만든다.\n"
    "각 문장마다 그 문장에 **실제로 들어있는 값 그대로** 정답을 함께 적는다 — "
    "문장에 없는 값은 정답도 null이어야 한다(지어내지 않는다).\n"
    "출력은 JSON 배열만. 각 원소:\n"
    '  {"question": str, "species": "dog"|"cat"|"bird"|null, '
    '"weight_kg": 숫자|null, "amount_g": 숫자|null, "substance": str|null, '
    '"elapsed_hours": 숫자|null}\n'
    "`elapsed_hours` 는 먹은 뒤 지난 시간을 **시간 단위 숫자**로 환산한 값이다 "
    "(예: '30분 전' → 0.5, '어제 저녁' 처럼 숫자화 안 되면 null).\n"
    "설명·코드블록 없이 배열만 출력한다."
)


def _gen_prompt(pattern: str, n: int, avoid: list[str]) -> str:
    avoid_block = ""
    if avoid:
        avoid_block = "\n\n이미 만든 문장(반복 금지):\n" + "\n".join(f"- {q}" for q in avoid[-15:])
    return f"패턴: {_PATTERNS[pattern]}\n이 패턴에 맞는 문장을 {n}개 만들어라.{avoid_block}"


def _parse_json_array(raw: str) -> list[dict]:
    m = re.search(r"\[.*\]", raw.strip(), re.DOTALL)
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
            sp = item.get("species")
            out.append(
                {
                    "question": str(item["question"]).strip(),
                    "species": sp if sp in _SPECIES else None,
                    "weight_kg": _num(item.get("weight_kg")),
                    "amount_g": _num(item.get("amount_g")),
                    "substance": (
                        str(item["substance"]).strip() if item.get("substance") else None
                    ),
                    "elapsed_hours": _num(item.get("elapsed_hours")),
                }
            )
    return out


def _num(v: object) -> float | None:
    try:
        return float(v) if v is not None and str(v).strip() != "" else None
    except (TypeError, ValueError):
        return None


def generate_for_pattern(pattern: str, target: int, batch_size: int, seen: set[str]) -> list[dict]:
    from pettriage.models.serving.factory import get_client

    client = get_client()
    if client is None:
        raise RuntimeError("LLM 클라이언트가 없다 — OPENAI_API_KEY 확인할 것")

    out: list[dict] = []
    recent: list[str] = []
    max_rounds = (target // batch_size + 1) * 3
    rounds = 0
    while len(out) < target and rounds < max_rounds:
        rounds += 1
        want = min(batch_size, target - len(out))
        raw = client.run_raw(_GEN_SYSTEM, _gen_prompt(pattern, want, recent), max_tokens=1800)
        items = _parse_json_array(raw)
        added = 0
        for item in items:
            q = item["question"]
            if not q or q in seen:
                continue
            seen.add(q)
            item["pattern"] = pattern
            out.append(item)
            recent.append(q)
            added += 1
        recent = recent[-15:]
        print(f"  [{pattern}] round {rounds}: +{added} (누적 {len(out)}/{target})")
        if added == 0 and not items:
            print(f"  ⚠ [{pattern}] 파싱 실패 — 원문 앞부분: {raw[:120]!r}")
    if len(out) < target:
        print(f"⚠ {pattern}: {target}건 목표, {len(out)}건만 생성됨")
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


def label_with_teacher(question: str) -> dict | None:
    """실제 프로덕션 SLOT 프롬프트로 라벨링한다 — `graph.nodes.slots._llm_slots` 그대로."""
    from pettriage.graph.nodes.slots import _llm_slots

    return _llm_slots(question)


def _slots_equal(a: dict, b: dict | None) -> bool:
    if b is None:
        return False
    keys = ("species", "weight_kg", "amount_g", "substance", "elapsed_hours")
    return all(a.get(k) == b.get(k) for k in keys)


def main() -> int:
    ap = argparse.ArgumentParser(description="② 슬롯 추출 distillation — LLM 생성판")
    ap.add_argument("--target", type=int, default=100, help="패턴당 목표 건수")
    ap.add_argument("--batch-size", type=int, default=15)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "slot_batch.jsonl")
    ap.add_argument(
        "--patterns",
        type=str,
        default=None,
        help="쉼표로 구분한 패턴 이름만 생성(예: elapsed_hours_mentioned). "
        "지정하면 --out에 이어쓴다(append) — 기존 건 안 건드림. 안 주면 전체 패턴을 새로 쓴다(overwrite).",
    )
    args = ap.parse_args()

    patterns = list(_PATTERNS)
    append = False
    if args.patterns:
        wanted = [p.strip() for p in args.patterns.split(",") if p.strip()]
        unknown = [p for p in wanted if p not in _PATTERNS]
        if unknown:
            raise SystemExit(f"모르는 패턴: {unknown} — 가능한 값: {list(_PATTERNS)}")
        patterns = wanted
        append = True

    golden_questions = _load_goldenset_questions()
    seen: set[str] = set(golden_questions)
    if append and args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line)["question"])

    all_candidates: list[dict] = []
    for pattern in patterns:
        print(f"=== {pattern} 생성 시작 (목표 {args.target}) ===")
        got = generate_for_pattern(pattern, args.target, args.batch_size, seen)
        all_candidates.extend(got)

    print(f"\n생성 완료: 총 {len(all_candidates)}건 → 교사 라벨링(교차검증) 시작")

    rows_out = []
    mismatches = 0
    from pettriage.models.serving.factory import client_name

    teacher = client_name()
    for i, c in enumerate(all_candidates, start=1):
        teacher_slots = label_with_teacher(c["question"])
        gen_slots = {
            k: c[k] for k in ("species", "weight_kg", "amount_g", "substance", "elapsed_hours")
        }
        agree = _slots_equal(gen_slots, teacher_slots)
        if not agree:
            mismatches += 1
        rows_out.append(
            {
                "question": c["question"],
                "pattern": c["pattern"],
                "gen_slots": gen_slots,
                "teacher_slots": teacher_slots,
                "agree": agree,
                "teacher": teacher,
            }
        )
        if i % 100 == 0:
            print(f"  교차검증 {i}/{len(all_candidates)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a" if append else "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n생성+교차검증 {len(rows_out)}건 → {args.out}")
    print(f"생성 정답 vs 교사(LLM) 불일치: {mismatches}/{len(rows_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
