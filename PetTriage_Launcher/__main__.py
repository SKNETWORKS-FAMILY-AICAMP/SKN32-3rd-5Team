# -*- coding: utf-8 -*-
"""save the pet — 실행기.

    더블클릭  실행.bat  (또는 이 파일)
    명령      python PetTriage_Launcher

설치부터 시연까지 **목적별로 골라 실행**한다. 환경변수를 사람이 외우지 않는다.

## 이 파일이 하는 세 가지

  ① **가상환경으로 자기 자신을 다시 실행한다.**
     더블클릭하면 시스템 파이썬이 뜨고, 그러면 패키지가 없어 바로 죽는다.
     `.venv` 를 찾아 그것으로 재실행한다 - 사용자는 모르게 넘어간다.

  ② **상태를 맨 위에 보여 준다.**
     무엇이 안 돼 있는지 보이면 무엇을 눌러야 할지 물어볼 필요가 없다.

  ③ **창이 바로 닫히지 않게 한다.**
     더블클릭 실행은 오류가 나면 창이 순식간에 사라진다. 그러면 오류를 못 읽는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

REEXEC_FLAG = "PETTRIAGE_LAUNCHER_REEXEC"


def _reexec_into_venv() -> None:
    """가상환경 파이썬으로 갈아탄다. **한 번만** 한다 (무한 재귀 방지)."""
    from checks import ROOT, in_venv, venv_python

    if in_venv() or os.environ.get(REEXEC_FLAG):
        return
    vpy = venv_python()
    if not vpy.exists():
        return  # 아직 없다 - [1] 처음 설치로 안내된다
    env = {**os.environ, REEXEC_FLAG: "1"}
    raise SystemExit(
        subprocess.call([str(vpy), str(Path(__file__).resolve())], cwd=str(ROOT), env=env)
    )


def _utf8_console() -> None:
    if os.name == "nt":
        os.system("")  # ANSI 활성화
        subprocess.call("chcp 65001 > nul", shell=True)
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8")
            except Exception:
                pass


MENU = """
  ── 준비 ────────────────────────────────────────────────
   1  처음 설치            venv 생성 + 패키지               5~10분
   2  .env 설정            키 입력 · 검증 · 셸 충돌 확인
   3  벡터 인덱스 만들기    !! 임베딩 4~9GB 내려받음         10~30분
   4  환경 점검            doctor

  ── 실행 ────────────────────────────────────────────────
   5  시연 서버 켜기        모드 선택 · !! 실제 LLM 은 질의마다 비용 발생
   6  평가 돌리기          비교군 선택 · !! 일부는 LLM 비용 발생
   7  테스트               pytest

  ── 그 밖 ───────────────────────────────────────────────
   8  시연 DB 초기화        쌓인 데모 데이터 지우기
   9  자유질의 검증         골든셋 밖 20건 · !! LLM 비용 발생

   0  종료
"""


def main() -> int:
    _utf8_console()
    _reexec_into_venv()

    import checks
    import steps

    actions = {
        "1": steps.install,
        "2": steps.setup_env,
        "3": steps.build_index,
        "4": steps.doctor,
        "5": steps.demo,
        "6": steps.evaluate,
        "7": steps.tests,
        "8": steps.reset_db,
        "9": steps.freeform,
    }

    while True:
        print("\n" + "=" * 72)
        print("  save the pet — 실행기")
        print("=" * 72)
        print("  " + checks.status_line())
        print(MENU)
        print("-" * 72)
        try:
            sel = input("  고르세요: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if sel == "0":
            return 0
        if sel not in actions:
            print("  1~9 또는 0 을 넣으세요.")
            continue

        stop = checks.blocking(sel)
        if stop:
            print(f"\n  !! {stop}")
            input("\n  엔터를 누르면 메뉴로 돌아갑니다 ")
            continue

        print("=" * 72)
        rc = actions[sel]()
        print("\n" + "=" * 72)
        print("  끝났습니다." if rc == 0 else f"  !! 종료 코드 {rc} - 위 메시지를 읽으세요.")
        input("  엔터를 누르면 메뉴로 돌아갑니다 ")


if __name__ == "__main__":
    try:
        code = main()
    except SystemExit as e:
        raise
    except Exception:
        import traceback

        traceback.print_exc()
        print("\n  예상치 못한 오류입니다. 위 내용을 그대로 알려 주세요.")
        input("  엔터를 누르면 닫힙니다 ")
        code = 1
    raise SystemExit(code)
