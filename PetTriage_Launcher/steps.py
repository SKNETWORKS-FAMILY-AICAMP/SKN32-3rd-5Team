# -*- coding: utf-8 -*-
"""메뉴 항목이 실제로 하는 일.

**공통 원칙 둘.**

  ① 환경변수를 사람이 외우게 하지 않는다.
     `PETTRIAGE_PROFILE=eval` 을 잊으면 `default.yaml` 의 `serve.engine: stub` 이
     답한다 — 겉보기에 잘 도는데 실제로는 아무것도 안 하는 상태다. 여기서 세운다.

  ② 실패를 삼키지 않는다.
     종료 코드를 그대로 보여 준다. 조용히 넘어가면 다음 단계에서 이유 모를 오류가 난다.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

from checks import ROOT, venv_python

PKG_EXTRAS = ".[api,rag,ingest,db,dev]"


def _run(args: list[str], *, env_extra: dict[str, str] | None = None, cwd: Path = ROOT) -> int:
    env = {**os.environ, **(env_extra or {})}
    print(f"\n  $ {' '.join(str(a) for a in args)}\n")
    try:
        return subprocess.call([str(a) for a in args], cwd=str(cwd), env=env)
    except KeyboardInterrupt:
        print("\n  (중단됨)")
        return 130
    except FileNotFoundError as e:
        print(f"  실행 파일을 찾지 못했습니다: {e}")
        return 127


def _py() -> str:
    """가상환경 파이썬. 없으면 지금 돌고 있는 것."""
    v = venv_python()
    return str(v) if v.exists() else sys.executable


def _eval_env() -> dict[str, str]:
    """평가·시연 프로필. **이걸 잊으면 stub 이 답한다.**"""
    return {"PETTRIAGE_PROFILE": "eval", "PYTHONUTF8": "1"}


# ── 1 처음 설치 ──────────────────────────────────────────────


def install() -> int:
    if not venv_python().exists():
        print("  가상환경을 만듭니다 (.venv)")
        rc = _run([sys.executable, "-m", "venv", ".venv"])
        if rc != 0:
            return rc
    print("  패키지를 설치합니다. 5~10분 걸립니다.")
    print("  [db] 를 포함합니다 - 없으면 인증 테스트 25건이 모듈째 건너뛰어집니다.")
    return _run([_py(), "-m", "pip", "install", "-e", PKG_EXTRAS, "-c", "constraints.txt"])


# ── 2 .env ───────────────────────────────────────────────────


def setup_env() -> int:
    return _run([_py(), "scripts/setup_env.py"])


# ── 3 벡터 인덱스 ────────────────────────────────────────────


def build_index() -> int:
    chroma = ROOT / ".chroma"
    if chroma.exists() and sum(1 for _ in chroma.rglob("*") if _.is_file()) > 3:
        print("  인덱스가 이미 있습니다.")
        if input("  다시 만들까요? [y/N] ").strip().lower() != "y":
            return 0
    print("  처음이면 임베딩 모델(BAAI/bge-m3)을 내려받습니다 - 4~9GB, 10~30분.")
    print("  네트워크와 디스크를 확보하고 시작하세요.")
    if input("  계속할까요? [Y/n] ").strip().lower() == "n":
        return 0
    return _run([_py(), "scripts/build_index.py", "--store", "chroma"], env_extra=_eval_env())


# ── 4 환경 점검 ──────────────────────────────────────────────


def doctor() -> int:
    return _run([_py(), "scripts/doctor.py"], env_extra=_eval_env())


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket() as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def _wait_until_up(host: str, port: int, proc: subprocess.Popen, timeout: int = 300) -> bool:
    """포트가 열릴 때까지 기다린다. **프로세스가 죽으면 즉시 그만둔다.**"""
    t0 = time.time()
    dots = 0
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            return False
        if _port_open(host, port, 0.4):
            return True
        time.sleep(0.6)
        dots += 1
        if dots % 10 == 0:
            print(f"  … 기다리는 중 ({int(time.time() - t0)}초)")
    return False


# ── 5 시연 서버 ──────────────────────────────────────────────


def demo() -> int:
    # **비용이 나가는 자리다.** 질의 1건마다 LLM 을 6~7회 부른다 —
    # 5태스크(분류·슬롯·압축·검증·평이화) + 초안 + 트리아지 판정.
    # 발표장에서 창을 켜 둔 채 여러 사람이 만지면 그만큼 늘어난다.
    print("\n  ── 어떤 모드로 띄울까요 ──────────────────────")
    print("   1  실제 LLM      진짜 답변           !! 질의마다 비용 발생")
    print("   2  무료 (폴백)   화면·흐름 확인용     비용 0 · 답변 품질은 낮다")
    print("   0  뒤로")
    sel = input("\n  고르세요: ").strip()
    if sel not in ("1", "2"):
        return 0
    free = sel == "2"
    if free:
        print("\n  무료 모드입니다. LLM 을 부르지 않고 코드·규칙만으로 답합니다.")
        print("  등급·근거·되묻기 흐름은 그대로 보이지만 문장은 자료 원문에 가깝습니다.")
    else:
        print("\n  !! 질의 1건마다 LLM 호출 6~7회 — 비용이 나갑니다.")
        print("     시연이 끝나면 Ctrl+C 로 반드시 끄세요.")
        if input("  계속할까요? [y/N] ").strip().lower() != "y":
            return 0

    db = ROOT / "pettriage.sqlite3"
    if not db.exists() or db.stat().st_size == 0:
        print("  DB 테이블이 없습니다. 먼저 만듭니다.")
        rc = _run([_py(), "-m", "pettriage.app.database"], env_extra=_eval_env())
        if rc != 0:
            print("  테이블 생성에 실패했습니다. [2] 로 .env 를 확인하세요.")
            return rc

    host, port = "127.0.0.1", 8000
    if _port_open(host, port, 0.3):
        print(f"  !! 포트 {port} 가 이미 사용 중입니다.")
        print("     이전 서버가 아직 떠 있을 수 있습니다. 그 창에서 Ctrl+C 로 끄세요.")
        return 1

    url = f"http://{host}:{port}"
    print(f"  서버를 켭니다 - {url}")
    print("  --reload 는 붙이지 않습니다 (윈도우에서 Ctrl+C 가 안 먹습니다).")
    print(f"  모드: {'무료(폴백) · 비용 0' if free else '실제 LLM · 질의마다 비용'}")
    print("  !! 첫 기동은 임베딩 모델을 읽느라 수십 초 걸립니다. 준비되면 브라우저가 열립니다.")
    print("  끄려면 Ctrl+C 를 누르세요.\n")

    extra = _eval_env()
    if free:
        # `--arm none` 과 같은 상태 — 그래프는 그대로 돌고 5태스크만 폴백으로 간다.
        # `engine=stub` 으로 내리는 것과 다르다. 그건 파이프라인 자체가 가짜다.
        extra["PETTRIAGE__MODEL__PROVIDER"] = "none"
    env = {**os.environ, **extra}
    proc = subprocess.Popen(
        [_py(), "-m", "uvicorn", "pettriage.app.main:app", "--host", host, "--port", str(port)],
        cwd=str(ROOT),
        env=env,
    )
    try:
        # **2.5초 기다리고 여는 방식은 틀렸다.** 모델 로딩에 수십 초가 걸리면
        # 브라우저가 "연결할 수 없습니다"를 띄우고, 사용자는 서버가 죽은 줄 안다.
        # 포트가 실제로 열릴 때까지 기다린다 — 프로세스가 죽으면 즉시 그만둔다.
        if _wait_until_up(host, port, proc, timeout=300):
            print(f"\n  준비됐습니다 - {url} 를 엽니다.\n")
            webbrowser.open(url)
        elif proc.poll() is not None:
            print(f"\n  !! 서버가 시작하지 못했습니다 (종료 코드 {proc.returncode}).")
            print("     위 로그를 읽으세요. [4] 환경 점검이 도움이 됩니다.")
            return proc.returncode or 1
        else:
            print("\n  !! 5분이 지나도 응답이 없습니다. 위 로그를 확인하세요.")
        proc.wait()
    except KeyboardInterrupt:
        print("\n  서버를 끕니다.")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    return 0


# ── 6 평가 ───────────────────────────────────────────────────

ARMS = [
    ("none", "코드·규칙만", "무료 · 1분 미만"),
    ("A", "직접 API", "LLM 60건 · 약 6분 · 비용 발생"),
    ("A-LC", "LangChain", "LLM 60건 · 약 6분 · 비용 발생"),
]


def evaluate() -> int:
    print("\n  ── 비교군 ──────────────────────────────────")
    for i, (arm, desc, cost) in enumerate(ARMS, 1):
        mark = "   " if arm == "none" else "!! "
        print(f"   {i}  {arm:6} {desc:14} {mark}{cost}")
    print("   0  뒤로")
    sel = input("\n  고르세요: ").strip()
    if sel == "0" or not sel.isdigit() or not (1 <= int(sel) <= len(ARMS)):
        return 0
    arm, _, cost = ARMS[int(sel) - 1]
    if arm != "none":
        print(f"\n  !! {cost}")
        if input("  계속할까요? [y/N] ").strip().lower() != "y":
            return 0
    stamp = time.strftime("%Y-%m-%d_%H%M")
    out = f"eval/reports/{stamp}_{arm}.json"
    return _run(
        [_py(), "eval/harness/run_eval.py", "--arm", arm, "--json", out],
        env_extra=_eval_env(),
    )


# ── 7 테스트 ─────────────────────────────────────────────────


def tests() -> int:
    print("  pytest 에 -q 를 붙이지 않습니다 - addopts 와 겹쳐 개수가 안 보입니다.")
    return _run([_py(), "-m", "pytest"])


# ── 8 DB 초기화 ──────────────────────────────────────────────


def reset_db() -> int:
    db = ROOT / "pettriage.sqlite3"
    if db.exists():
        print(f"  {db.name} 를 지웁니다 ({db.stat().st_size // 1024}KB)")
        if input("  시연 데이터가 모두 사라집니다. 계속할까요? [y/N] ").strip().lower() != "y":
            return 0
        try:
            db.unlink()
        except OSError as e:
            print(f"  지우지 못했습니다: {e}  (서버가 켜져 있으면 먼저 끄세요)")
            return 1
    return _run([_py(), "-m", "pettriage.app.database"], env_extra=_eval_env())


# ── 9 자유질의 ───────────────────────────────────────────────


def freeform() -> int:
    print("  골든셋 밖 질문 20건을 돌립니다.")
    print("  !! 실제 LLM 을 부릅니다 - 3~5분, 비용 발생.")
    if input("  계속할까요? [y/N] ").strip().lower() != "y":
        return 0
    stamp = time.strftime("%Y-%m-%d_%H%M")
    return _run(
        [_py(), "scripts/probe_freeform.py", "--json", f"eval/reports/자유질의_{stamp}.json"],
        env_extra=_eval_env(),
    )
