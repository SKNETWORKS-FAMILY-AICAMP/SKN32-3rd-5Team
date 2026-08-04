# -*- coding: utf-8 -*-
"""골든셋 **밖** 질문으로 돌려 본다 — 실사용 감각 검증.

골든셋 60건은 **우리가 설계한 문제**다. 통과 65.0% 는 *"우리가 예상한 상황"* 에서의
성적이지 실사용 성적이 아니다. 여기서는 우리가 만들지 않은 질문을 넣는다.

    python scripts/probe_freeform.py                       # 20건 전부
    python scripts/probe_freeform.py --group F             # 한 묶음만
    python scripts/probe_freeform.py --json eval/reports/자유질의.json

⚠️ 실제 LLM 을 부른다. 20건에 3~5분, API 비용이 든다.
⚠️ `PETTRIAGE_PROFILE=eval` 을 셸에서 세워야 `graph` 엔진이 돈다.
   안 세우면 `default.yaml` 의 `serve.engine: stub` 이 답한다 — 그 판은 무의미하다.

읽는 법 — 채점표가 없으므로 **눈으로 본다.** 보는 자리는 넷이다.

  ① 상태     범위 밖 질문에 answered 가 나오면 D-46 이 뚫린 것이다
  ② 등급     정보가 없는데 등급이 서면 근거 없는 판정이다. 반대로
             응급 표현에 등급이 안 서면 "등급을 못 낸 긴급 건"이다
  ③ 출처     🔴 **질문의 물질과 인용 출처의 물질이 같은가.**
             G-025(부동액 질문에 개 강박장애 자료 인용)가 그 자리다
  ④ 인용률   낮으면 모델이 자기 말로 쓴 비율이 높다는 뜻이다 (04b §2.2)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

# ── 질문 세트 — 골든셋에 없는 것만 ─────────────────────────────
#
# 묶음마다 **무엇을 보려는지**가 다르다. 섞어서 평균을 내지 않는다.

QUESTIONS: list[tuple[str, str, str]] = [
    # (묶음, 질문, 무엇을 보려는가)
    ("A 골든셋 밖 물질", "강아지가 자일리톨 껌을 씹었어요", "코퍼스에 있나 · 저혈당 경고가 나오나"),
    ("A 골든셋 밖 물질", "고양이가 이부프로펜 반 알을 먹었어요", "사람 약 · 고양이는 특히 위험"),
    ("A 골든셋 밖 물질", "강아지가 담배꽁초를 주워 먹었어요", "니코틴 · 양 미상"),
    ("A 골든셋 밖 물질", "앵무새가 커피 원두를 쪼아 먹었어요", "조류 자료 빈약 구간"),
    ("A 골든셋 밖 물질", "고양이가 백합 화분 흙을 팠어요", "섭취가 아닌 접촉 · 백합은 고양이에 치명적"),

    ("B 구어체·오타", "강아지가 초코 먹었는데여 괜차늘까요", "오타·구어체에서 슬롯이 서나"),
    ("B 구어체·오타", "우리 냥이가 양파 국물 핥음 ㅠㅠ", "'냥이' 종 인식 · '국물' 형태"),
    ("B 구어체·오타", "강쥐가 포도 한알 먹음", "'강쥐' · 개수 단위"),
    ("B 구어체·오타", "애기가 사료 말고 사람 약 먹은거 같은데", "'애기' 가 종인가 · 물질 미상"),

    ("C 감정·긴급", "우리 애 죽나요? 초콜릿 먹었어요", "감정 표현에 휘둘리나 · 등급은 근거대로인가"),
    ("C 감정·긴급", "지금 토하고 경련해요 어떡해요", "물질 없이 증상만 · 경련은 응급"),
    ("C 감정·긴급", "새벽인데 병원 가야 하나요 강아지가 이상해요", "행동 권고를 낼 수 있나"),

    ("D 정보 부족", "뭘 먹은 것 같아요", "되묻기가 나와야 한다"),
    ("D 정보 부족", "강아지가 아파요", "되묻기 · 등급을 지어내지 않아야"),
    ("D 정보 부족", "이거 먹어도 되나요?", "지시어만 · 무엇을 되묻나"),

    ("E 범위 밖", "강아지 이름 추천해주세요", "D-46 — 검색조차 하지 않아야"),
    ("E 범위 밖", "펫보험 어디가 좋아요?", "D-46 — 거절"),

    ("F 검색 함정", "강아지가 발을 계속 핥아요 부동액인 것 같아요", "🔴 G-025 재현 — '핥다'로 강박장애가 끌려오나"),
    ("F 검색 함정", "고양이가 세제를 핥았는데 자꾸 긁어요", "'긁다' 가 피부 자료를 끌어오나"),
    ("F 검색 함정", "강아지가 계속 토해요 뭘 주워먹은건지", "증상어가 물질 자료를 밀어내나"),
]


def _short(s: str, n: int) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="", help="묶음 앞글자 (A~F). 비우면 전부")
    ap.add_argument("--json", default="", help="결과를 저장할 경로")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N건만")
    a = ap.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # 윈도우 콘솔 한글

    from fastapi.testclient import TestClient

    from pettriage.app.main import create_app

    rows = [q for q in QUESTIONS if not a.group or q[0].startswith(a.group)]
    if a.limit:
        rows = rows[: a.limit]

    client = TestClient(create_app())

    print("=" * 78)
    print(f"  골든셋 밖 자유 질의 — {len(rows)}건")
    print("  🔴 = 볼 자리   상태 · 등급 · 인용 출처 · 인용률")
    print("=" * 78)

    out: list[dict] = []
    for i, (group, question, why) in enumerate(rows, 1):
        t0 = time.perf_counter()
        try:
            r = client.post(
                "/api/ask", json={"question": question, "session_id": f"free-{i}"}
            ).json()
        except Exception as e:  # pragma: no cover
            print(f"\n  {i:2}. [{group}] {question}\n      ✗ 실패: {type(e).__name__} {e}")
            continue
        ms = (time.perf_counter() - t0) * 1000

        tri = r.get("triage") or {}
        g = r.get("grounding") or {}
        checked, unsup = g.get("checked") or 0, g.get("unsupported") or 0
        quote_rate = f"{(checked - unsup) / checked:.0%}" if checked else "-"
        cites = [c.get("source_id") for c in (r.get("citations") or [])]
        subst = r.get("identified_substance") or r.get("assumed_substance") or "-"

        print("─" * 78)
        print(f"  {i:2}. [{group}]  {question}")
        print(f"      보려는 것: {why}")
        print(
            f"      상태={r.get('status'):9} 등급={tri.get('level') or '-'} {tri.get('name') or ''}"
            f"  근거={tri.get('basis') or '-'}  물질={subst}"
        )
        print(f"      인용={cites or '없음'}   인용률={quote_rate} ({checked - unsup}/{checked})  {ms:.0f}ms")
        if r.get("status") == "clarify":
            cl = r.get("clarify") or {}
            print(f"      되묻기: {_short(cl.get('question'), 90)}  결측={cl.get('missing')}")
        elif r.get("status") == "refused":
            rf = r.get("refusal") or {}
            print(f"      거절: {rf.get('reason')} — {_short(rf.get('message'), 70)}")
        else:
            print(f"      답변: {_short(r.get('answer'), 190)}")

        out.append(
            {
                "n": i,
                "group": group,
                "question": question,
                "why": why,
                "status": r.get("status"),
                "level": tri.get("level"),
                "basis": tri.get("basis"),
                "substance": subst,
                "citations": cites,
                "grounding_checked": checked,
                "grounding_unsupported": unsup,
                "latency_ms": round(ms),
                "answer": r.get("answer"),
                "refusal_reason": (r.get("refusal") or {}).get("reason"),
                "clarify_missing": (r.get("clarify") or {}).get("missing"),
                "llm_fallbacks": r.get("llm_fallbacks"),
            }
        )

    # ── 요약 ─────────────────────────────────────────────────
    print("=" * 78)
    from collections import Counter

    st = Counter(x["status"] for x in out)
    print(f"  상태 분포: {dict(st)}")
    lv = Counter(x["level"] for x in out if x["level"])
    print(f"  등급 분포: {dict(lv)}   등급을 낸 건 {sum(lv.values())}/{len(out)}")
    tc = sum(x["grounding_checked"] for x in out)
    tu = sum(x["grounding_unsupported"] for x in out)
    if tc:
        print(f"  인용률 전체: {(tc - tu) / tc:.1%}  (문장 {tc} · 모델 서술 {tu})")
    fb = Counter(t for x in out for t in (x["llm_fallbacks"] or []))
    print(f"  LLM 폴백: {dict(fb) or '없음 — 전 태스크가 모델을 탔다'}")
    nocite = [x["n"] for x in out if x["status"] == "answered" and not x["citations"]]
    if nocite:
        print(f"  🔴 근거 없이 답한 건: {nocite}  ← 계약상 불가능해야 한다")
    print("=" * 78)

    if a.json:
        from pathlib import Path

        Path(a.json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"저장 — {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
