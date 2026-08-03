#!/usr/bin/env python3
"""LLM 클라이언트 스모크 테스트 — **평가를 돌리기 전에 여기서 걸러낸다.**

    python scripts/smoke_llm.py --arm none   # 기준선 — 어디서나 돈다
    python scripts/smoke_llm.py --arm A      # 대형 LLM
    python scripts/smoke_llm.py --arm D      # Qwen3-4B 베이스
    python scripts/smoke_llm.py --arm C      # Qwen3-4B + 어댑터
    python scripts/smoke_llm.py --arm D --call   # 실제로 1회 호출 (가중치를 받는다)

`--arm` 을 안 주면 지금 설정 그대로 본다.

설계 근거: 04 §8 · D-65 · D-69

⚠️ **`LocalQwenClient` 는 2026-08-02 까지 한 번도 실행된 적이 없다.**

    노드 4곳이 `APIClient()` 를 하드코딩하고 있어서 아무도 그것을 만들지 않았고,
    그래서 `run_raw` 가 통째로 빠져 있는 것도 드러나지 않았다 (D-65).
    골든셋 60건 × LLM 6회를 돌리다가 **첫 호출에서** 터지면 받은 시간이 통째로 날아간다.

    이 스크립트는 **비싼 것을 하기 전에 싼 것부터** 확인한다 —
    ① 설정이 무엇을 가리키나 → ② 필요한 패키지가 있나 → ③ (선택) 실제 호출.

`--call` 없이는 **가중치를 받지 않는다.** 환경 점검만 한다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _mark(ok: bool) -> str:
    return "✅" if ok else "❌"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=["none", "A", "A-LC", "C", "D"], help="04 §3 비교군 (arms.py)")
    ap.add_argument("--call", action="store_true", help="실제로 1회 호출한다 (가중치를 받는다)")
    a = ap.parse_args()

    if a.arm:
        from pettriage.models.serving.arms import NEEDS, apply_arm

        applied = apply_arm(a.arm)
        print(f"── 비교군 {a.arm} ─────────────────────────────────")
        for k, v in applied.items():
            print(f"  {k}={v}")
        print(f"  필요: {NEEDS[a.arm]}\n")

    from pettriage.config import get_config
    from pettriage.models.serving.factory import client_name, get_client

    m = get_config().model
    print("── 설정 ─────────────────────────────────────────")
    print(f"  provider      {m.provider}")
    if m.provider in ("api", "langchain"):
        if m.provider == "langchain":
            print("  연동          LangChain (langchain_openai.ChatOpenAI) — D-71")
        print(f"  api_model     {m.api_model}")
        print(f"  api_base_url  {m.api_base_url or '(OpenAI 본가)'}")
    elif m.provider == "qwen":
        print(f"  base_id       {m.base_id}")
        print(f"  revision      {m.revision or '(없음 — 최신을 받는다. 재현이 깨진다)'}")
        print(f"  dtype         {m.dtype}   load_in_4bit={m.load_in_4bit}")
        # **C 와 D 를 가르는 유일한 값이다.** 안 보이면 어느 쪽을 잰 건지 모른다.
        print(f"  adapter_path  {m.adapter_path or '(없음 → 베이스 = 비교군 D)'}")

    print("\n── 환경 ─────────────────────────────────────────")
    cuda = False
    if m.provider == "qwen":
        try:
            import torch

            cuda = torch.cuda.is_available()
            name = torch.cuda.get_device_name(0) if cuda else "CPU"
            print(f"  {_mark(True)} torch {torch.__version__}  · {name}")
            if not cuda:
                print("     ⚠ GPU 가 없다 — 4B 생성은 매우 느리다. 8건 프로브만 권한다.")
        except ImportError:
            print(f"  {_mark(False)} torch 없음 — pip install -e '.[train]' 또는 pip install torch")
            return 1
        for pkg, why in (
            ("accelerate", "device_map='auto' 에 필요"),
            ("bitsandbytes", "load_in_4bit 에 필요 (4bit 가중치를 읽으려면 필수)"),
        ):
            try:
                __import__(pkg)
                print(f"  {_mark(True)} {pkg}")
            except ImportError:
                # 4bit 가중치를 쓰기로 해 놓고 bitsandbytes 가 없으면 **못 읽는다** — ❌.
                # accelerate 는 없어도 폴백하므로 경고(⚠)로 둔다. 둘을 같은 표시로
                # 두면 "무엇을 반드시 깔아야 하나" 가 안 보인다.
                blocking = m.load_in_4bit and pkg == "bitsandbytes"
                print(f"  {'❌' if blocking else '⚠'} {pkg} 없음 — {why}")
                if blocking:
                    print("     → pip install bitsandbytes accelerate")

    client = get_client()
    print(f"\n  클라이언트  {client_name()}")
    if client is None:
        print("  ⚠ 모델 없이 돈다 — 5태스크가 전부 폴백이다 (04 §3 기준선).")
        return 0

    # **프로토콜을 다 갖췄나.** `run_raw` 가 빠져 있던 전례가 있다 (D-65).
    missing = [n for n in ("run", "run_raw") if not hasattr(client, n)]
    print(f"  {_mark(not missing)} 프로토콜 {'완비' if not missing else f'누락 {missing}'}")
    if missing:
        return 1

    if not a.call:
        print("\n여기까지가 가중치 없이 되는 확인이다. 실제 호출은 `--call`.")
        return 0

    print("\n── 호출 1회 ─────────────────────────────────────")
    print("  (처음이면 가중치를 받는다. 몇 분 걸릴 수 있다)")
    from pettriage.models.tasks import Task

    q = "강아지가 초콜릿을 먹었어요"
    try:
        out = client.run(Task.CLASSIFY, q, max_tokens=16)
        print(f"  {_mark(True)} run(CLASSIFY)  {q!r} → {out.strip()[:60]!r}")
    except Exception as e:  # noqa: BLE001 — 스모크 테스트다. 무엇이든 보여준다
        print(f"  {_mark(False)} run 실패 — {type(e).__name__}: {e}")
        return 1
    try:
        raw = client.run_raw("한 문장으로 답한다.", q, max_tokens=40)
        print(f"  {_mark(True)} run_raw        → {raw.strip()[:60]!r}")
    except Exception as e:  # noqa: BLE001
        print(f"  {_mark(False)} run_raw 실패 — {type(e).__name__}: {e}")
        return 1

    print("\n✅ 통과. 이제 프로브를 돌린다 —")
    print(
        "  python eval/harness/run_eval.py --only G-001 G-028 G-029 G-030 "
        "G-040 G-041 G-011 G-017 --json eval/reports/probe.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
