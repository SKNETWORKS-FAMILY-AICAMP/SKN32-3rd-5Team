#!/usr/bin/env python
"""① 분류 태스크 — distillation 파일럿.

설계 근거: docs/03_모델-멀티태스크학습.md §4 · docs/03a_파인튜닝-구현기획.md §3

    보호자 질의 생성(템플릿 증강, 비용 없음) → 대형 LLM 라벨링(distillation,
    실제 프로덕션 CLASSIFY 프롬프트 재사용) → 사람 검수용 파일 출력.

    **이 스크립트는 `data/train/samples.jsonl` 를 직접 쓰지 않는다.** 03이
    요구하는 "대형 LLM 라벨링 → 사람 검수" 중 검수는 사람이 해야 하므로,
    검수 전 후보만 `--out` 에 낸다. 검수 후 `--commit` 으로 반영한다.

    라벨은 `graph.nodes.classify.ALLOWED_INTENTS` 4종 그대로 쓴다 —
    더미 샘플의 "중독_고위험" 같은 의도+위험도 합성 라벨은 **쓰지 않는다.**
    지금 배포된 classify.py 가 그 라벨을 만들지 않으므로, 다른 형식으로
    학습하면 추론 시점과 어긋난다 (05 §4 "학습과 추론이 같은 문자열을 쓴다").

    골든셋과 겹치는 질문은 만들지 않는다 — 학습·평가 누수 방지(D-29).

    python scripts/generate_classify_samples.py --n 30 --out data/train/classify_pilot.jsonl
    python scripts/generate_classify_samples.py --n 30 --commit data/train/samples.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

#: 종별로 실제 코퍼스에 있는, 자연스러운 질문을 만들 수 있는 물질만 골랐다.
#: `compute.vocabulary` 533종 전체를 쓰지 않는 이유 — 상당수가
#: "Albright's Chicken Recipe(리콜)" 같은 제품명이라 보호자 발화로 안 나온다.
_INTOXICATION_SUBSTANCES: dict[str, tuple[str, ...]] = {
    "dog": ("초콜릿", "포도", "양파", "마늘", "자일리톨", "아보카도", "커피", "부동액", "건포도"),
    "cat": ("백합", "초콜릿", "양파", "알로에", "국화", "포도"),
    "bird": ("아보카도", "초콜릿", "PTFE(프라이팬 흄)", "카페인"),
}

#: `{species}`·`{species_이}`·`{species_을}` 자리를 쓴다 — 조사가 붙는 자리는
#: 템플릿에서부터 명시해, `{species}` 뒤에 조사를 잘못 이어붙이는 실수를 막는다.
_INTOXICATION_TEMPLATES = (
    "{species_이} {substance_을} 먹었는데 괜찮을까요",
    "{species_이} {substance_을} 조금 핥아먹은 것 같아요",
    "산책하다가 {species_이} {substance_을} 주워 먹었어요",
    "{species}한테 실수로 {substance_을} 줬는데 위험한가요",
)

_SYMPTOM_TEMPLATES = (
    "{species_이} 계속 구토를 해요",
    "{species_이} 기운이 없고 축 처져 있어요",
    "{species_이} 갑자기 설사를 시작했어요",
    "{species_이} 몸을 떨면서 침을 흘려요",
)

_NUTRITION_TEMPLATES = (
    "{species} 사료는 하루에 얼마나 줘야 하나요",
    "{species}한테 간식으로 뭘 줘도 되나요",
    "{species} 체중 관리를 위해 급여량을 줄여도 될까요",
    "{species}한테 필요한 하루 칼슘 양이 얼마나 되나요",
)

_GENERAL_TEMPLATES = (
    "{species} 이름 추천 좀 해주세요",
    "{species} 미용 잘하는 곳 알려주세요",
    "{species} 보험료가 얼마나 하나요",
    "{species} 훈련 용품 추천해주세요",
)

_SPECIES_KO = {"dog": "강아지", "cat": "고양이", "bird": "앵무새"}


def _has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있나 — 을/를, 이/가 조사 선택에 쓴다."""
    ch = word[-1]
    if not ("가" <= ch <= "힣"):
        return False  # 한글 완성형이 아니면(영문 등) 받침 없는 쪽으로 취급
    return (ord(ch) - ord("가")) % 28 != 0


def _with_josa(noun: str, with_batchim: str, without_batchim: str) -> str:
    return f"{noun}{with_batchim}" if _has_batchim(noun) else f"{noun}{without_batchim}"


def _fill(tmpl: str, *, species: str, substance: str | None = None) -> str:
    return tmpl.format(
        species=species,
        species_이=_with_josa(species, "이", "가"),
        species_을=_with_josa(species, "을", "를"),
        substance_을=_with_josa(substance, "을", "를") if substance else "",
    )


#: 조합 공간(종×물질×템플릿, 종×템플릿)이 작아 **뽑다 보면 겹친다** —
#: n_per_intent 가 조합 수에 가까워질수록 흔해진다. 최대 시도 안에서
#: 중복을 걸러내고, 그래도 못 채우면 **있는 만큼만** 반환한다 — 지어내지 않는다.
_MAX_TRIES_FACTOR = 20


def _gen_intoxication(n: int, rng: random.Random, seen: set[str]) -> list[tuple[str, str | None]]:
    out = []
    species_list = list(_INTOXICATION_SUBSTANCES)
    tries = 0
    while len(out) < n and tries < n * _MAX_TRIES_FACTOR:
        tries += 1
        species = rng.choice(species_list)
        substance = rng.choice(_INTOXICATION_SUBSTANCES[species])
        tmpl = rng.choice(_INTOXICATION_TEMPLATES)
        q = _fill(tmpl, species=_SPECIES_KO[species], substance=substance)
        if q in seen:
            continue
        seen.add(q)
        out.append((q, species))
    return out


def _gen_other(
    templates: tuple[str, ...], n: int, rng: random.Random, seen: set[str]
) -> list[tuple[str, str | None]]:
    out = []
    species_list = [*_SPECIES_KO, None]  # None = 종 미명시 질문도 섞는다
    tries = 0
    while len(out) < n and tries < n * _MAX_TRIES_FACTOR:
        tries += 1
        species = rng.choice(species_list)
        tmpl = rng.choice(templates)
        noun = _SPECIES_KO.get(species, "반려동물")
        q = _fill(tmpl, species=noun)
        if q in seen:
            continue
        seen.add(q)
        out.append((q, species))
    return out


def generate_questions(n_per_intent: int, seed: int) -> list[dict]:
    """4개 의도 × n_per_intent 문항. **문장 전역에서 중복을 걸러낸다** —
    조합 공간이 작은 태스크(nutrition/general 등)는 요청한 개수를 못 채울 수 있다.
    """
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[dict] = []
    generators = (
        ("intoxication", lambda: _gen_intoxication(n_per_intent, rng, seen)),
        ("symptom", lambda: _gen_other(_SYMPTOM_TEMPLATES, n_per_intent, rng, seen)),
        ("nutrition", lambda: _gen_other(_NUTRITION_TEMPLATES, n_per_intent, rng, seen)),
        ("general", lambda: _gen_other(_GENERAL_TEMPLATES, n_per_intent, rng, seen)),
    )
    for intent, gen in generators:
        pairs = gen()
        if len(pairs) < n_per_intent:
            print(f"⚠ {intent}: 조합 공간 부족 — {n_per_intent}건 요청, {len(pairs)}건만 고유")
        for question, species in pairs:
            rows.append({"question": question, "species": species, "expected_intent": intent})
    rng.shuffle(rows)
    return rows


def _load_goldenset_questions() -> set[str]:
    """골든셋 질문 원문 집합 — 생성 질문이 겹치면 제외한다 (D-29 누수 방지)."""
    import csv

    out: set[str] = set()
    golden_dir = ROOT / "eval" / "goldenset"
    for path in sorted(golden_dir.glob("golden_*.csv")):
        with path.open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                q = (row.get("question") or "").strip()
                if q:
                    out.add(q)
    return out


def label_with_teacher(question: str) -> tuple[str, str]:
    """실제 프로덕션 CLASSIFY 프롬프트로 라벨링한다 (distillation 교사).

    `graph.nodes.classify._llm_classify` 를 그대로 재사용한다 — 학습 라벨을
    만드는 프롬프트와 추론 프롬프트가 다르면 나중에 성능 차이의 원인을
    영원히 못 찾는다 (models/prompts.py 상단 설계 근거와 같은 이유).

    Returns:
        (label, teacher_model_id). LLM이 없거나 실패하면 label="unknown".
    """
    from pettriage.graph.nodes.classify import ALLOWED_INTENTS, _llm_classify
    from pettriage.models.serving.factory import client_name

    raw = _llm_classify(question)
    label = raw if raw in ALLOWED_INTENTS else "unknown"
    return label, client_name()


def main() -> int:
    ap = argparse.ArgumentParser(description="① 분류 태스크 distillation 파일럿")
    ap.add_argument("--n", type=int, default=8, help="의도(4종)당 생성 개수 (기본 8 → 총 32건)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train" / "classify_pilot.jsonl")
    args = ap.parse_args()

    golden_questions = _load_goldenset_questions()
    candidates = generate_questions(args.n, args.seed)

    kept = [c for c in candidates if c["question"] not in golden_questions]
    dropped = len(candidates) - len(kept)
    if dropped:
        print(f"골든셋과 겹쳐 제외: {dropped}건")

    rows_out = []
    mismatches = 0
    for c in kept:
        label, teacher = label_with_teacher(c["question"])
        agree = label == c["expected_intent"]
        if not agree:
            mismatches += 1
        rows_out.append(
            {
                "question": c["question"],
                "species": c["species"],
                "template_intent": c["expected_intent"],
                "teacher_label": label,
                "agree": agree,
                "teacher": teacher,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"생성 {len(rows_out)}건 → {args.out}")
    print(f"템플릿 의도 vs 교사(LLM) 라벨 불일치: {mismatches}/{len(rows_out)}")
    print("불일치 건은 둘 중 하나가 틀렸다는 뜻이다 — 사람이 확인해야 한다 (03 §4 사람 검수).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
