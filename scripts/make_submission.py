#!/usr/bin/env python3
"""필수 산출물 넷을 `SKN_3rd_5Team_REPORT/` 한 폴더로 모은다.

    python scripts/make_submission.py             # 폴더 재생성
    python scripts/make_submission.py --dry-run   # 무엇을 복사할지만 본다
    python scripts/make_submission.py --zip       # 폴더 + SKN_3rd_5Team_REPORT_<해시>.zip

정리: 오한빈 (팀장)

    문서는 전부 이미 `docs/` · `data/` · `eval/` 에 있다. 이 폴더는 **제출물을 받는
    분이 필수 산출물 넷을 한자리에서 찾을 수 있도록** 모아 둔 것이다.

⚠️ **이 폴더를 손으로 고치지 않는다.**

    이 폴더는 **전부 사본**이다. 원본은 `docs/` · `data/` · `eval/` 에 있다.
    사본을 고치면 어느 쪽이 진짜인지 알 수 없게 된다 — 이 저장소가 데이터에
    적용한 것과 같은 원칙이다 (D-22 · 단일 출처).
    **무엇이 바뀌었으면 원본을 고치고 이 스크립트를 다시 돌린다.**

⚠️ **코드는 복사하지 않는다.**

    `src/` 를 복사하면 저장소 안에 두 벌의 코드가 생기고, 채점자는 어느 쪽을
    읽어야 하는지 알 수 없다. 대신 `03_구현-코드/README.md` 가
    **파일 경로와 재현 명령**을 적는다. 코드의 원본은 저장소 자체다.

⚠️ **이 폴더는 커밋하지 않는다** (`.gitignore`).

    안에 든 것이 전부 이미 저장소에 있다. 커밋하면 같은 내용이 두 번 들어가고,
    한쪽만 고쳐지는 순간 저장소가 거짓말을 시작한다. 제출은 `--zip` 으로 만든
    묶음을 올린다.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "SKN_3rd_5Team_REPORT"

#: 생성물 표시. 이 파일이 없는 같은 이름 폴더는 사람이 만든 것으로 보고 **지우지 않는다.**
MARKER = ".생성물"

#: 예전 이름들. 이름을 바꾸면 옛 폴더가 남아 **두 벌이 되므로** 여기에 적고 함께 치운다.
#: 표시 파일이 있는 것만 지운다 — 사람이 만든 같은 이름 폴더는 건드리지 않는다.
LEGACY_NAMES = ("제출",)

# ──────────────────────────────────────────────────────────────
# 무엇을 어디에 담는가
#
# 항목은 "저장소 상대 경로" 또는 ("저장소 상대 경로", "폴더 안 상대 경로").
# 두 번째 형태는 하위 폴더를 만들고 싶을 때만 쓴다.
# ──────────────────────────────────────────────────────────────

GROUPS: list[dict] = [
    {
        "dir": "01_데이터-전처리",
        "원본": "`docs/01*.md` · `data/facts/` · `data/manifests/`",
        "요구": "수집된 데이터 및 데이터 전처리 문서",
        "핵심": "01_데이터-수집및전처리.md",
        "한줄": "원문을 옮기지 않고 사실을 옮겼다 — 수집 전 게이트 · 사실 표 888행 · 삭제 이력",
        "items": [
            "docs/01_데이터-수집및전처리.md",
            "docs/01a_자료분석보고.md",
            "docs/01b_자료검증보고.md",
            "docs/01c_데이터-작업지시.md",
            "docs/01d_자료보관규칙.md",
            "docs/01e_사실표작성지침.md",
            ("data/README.md", "데이터폴더_README.md"),
            ("data/facts/facts_ohb.csv", "사실표/facts_ohb.csv"),
            ("data/facts/README.md", "사실표/README.md"),
            ("data/facts/검수목록_ohb.md", "사실표/검수목록_ohb.md"),
            ("data/facts/PR3_검수보고.md", "사실표/PR3_검수보고.md"),
            ("data/facts/사실표_양식.csv", "사실표/사실표_양식.csv"),
            ("data/manifests/MANIFEST.csv", "매니페스트/MANIFEST.csv"),
            ("data/manifests/SOURCES_CITED.csv", "매니페스트/SOURCES_CITED.csv"),
            ("data/manifests/DELETION_LOG.csv", "매니페스트/DELETION_LOG.csv"),
            ("data/manifests/SNAPSHOT_MANIFEST.csv", "매니페스트/SNAPSHOT_MANIFEST.csv"),
            ("src/pettriage/compute/tables/정량임계치.csv", "생성물/정량임계치.csv"),
            ("src/pettriage/compute/tables/정성등급.csv", "생성물/정성등급.csv"),
            ("src/pettriage/compute/tables/물질어휘.csv", "생성물/물질어휘.csv"),
            ("src/pettriage/compute/tables/별칭.csv", "생성물/별칭.csv"),
            ("src/pettriage/compute/tables/성분함량.csv", "생성물/성분함량.csv"),
            ("src/pettriage/compute/tables/README.md", "생성물/README.md"),
        ],
    },
    {
        "dir": "02_시스템-아키텍처",
        "원본": "`docs/02_시스템-아키텍처.md` · `docs/그림/` · `configs/`",
        "요구": "시스템 아키텍처",
        "핵심": "02_시스템-아키텍처.md",
        "한줄": "노드 18개 · 저장소 3종 · API 14개 — 구조도는 코드에서 뽑는다",
        "items": [
            "docs/02_시스템-아키텍처.md",
            "docs/시스템설계_한장.pdf",
            "docs/시스템설계_한장.html",
            ("docs/그림/질의그래프.mmd", "그림/질의그래프.mmd"),
            ("docs/그림/질의그래프.png", "그림/질의그래프.png"),
            ("docs/그림/질의그래프.pdf", "그림/질의그래프.pdf"),
            ("docs/그림/개념도1_사전바닥.png", "그림/개념도1_사전바닥.png"),
            ("docs/그림/개념도2_방어층_G-025.png", "그림/개념도2_방어층_G-025.png"),
            ("docs/그림/개념도3_코드와LLM분업.png", "그림/개념도3_코드와LLM분업.png"),
            ("docs/그림/개념도_원본.html", "그림/개념도_원본.html"),
            ("configs/default.yaml", "설정/default.yaml"),
            ("configs/eval.yaml", "설정/eval.yaml"),
        ],
    },
    {
        "dir": "03_구현-코드",
        "원본": "`src/pettriage/` · `tests/` (사본 없음 — 경로 지도만)",
        "요구": "개발된 소프트웨어 — RAG 기반 LLM과 벡터 데이터베이스 연동 구현 코드",
        "핵심": "README.md  (경로 지도 · 코드는 복사하지 않는다)",
        "한줄": "LangGraph 18노드 · Chroma 888청크 · LangChain(ChatOpenAI) 연동 · pytest 585",
        "items": [],  # README.md 만 생성한다 (아래 CODE_MAP)
    },
    {
        "dir": "04_테스트-결과보고서",
        "원본": "`docs/04*.md` · `eval/`",
        "요구": "테스트 계획 및 결과 보고서",
        "핵심": "04_테스트-평가계획.md · 2026-08-04_결과보고서.md",
        "한줄": "골든셋 60건 · 다섯 판 측정 · 통과 45.0% → 65.0% · 중대 과소평가 0건",
        "items": [
            "docs/04_테스트-평가계획.md",
            "docs/04a_골든셋작성지침.md",
            "docs/04b_발견_근거검증과출처분해.md",
            ("eval/reports/결과보고서_제출용/2026-08-04_결과보고서.md", "2026-08-04_결과보고서.md"),
            ("eval/reports/결과보고서_제출용/결과보고서_발표용.pdf", "결과보고서_발표용.pdf"),
            ("eval/reports/결과보고서_제출용/결과보고서_한장요약.html", "결과보고서_한장요약.html"),
            (
                "eval/reports/2026-08-03_결과보고서.md",
                "원자료/2026-08-03_결과보고서_WS3파인튜닝.md",
            ),
            ("eval/reports/D1.json", "원자료/D1_A_대형LLM.json"),
            ("eval/reports/D1_lc.json", "원자료/D1_A-LC_LangChain.json"),
            ("eval/reports/D1_none.json", "원자료/D1_none_기준선.json"),
            ("eval/README.md", "하네스/eval_README.md"),
            ("eval/harness/README.md", "하네스/harness_README.md"),
            ("eval/harness/run_eval.py", "하네스/run_eval.py"),
            ("eval/harness/metrics.py", "하네스/metrics.py"),
            ("eval/goldenset/README.md", "골든셋/README.md"),
            ("eval/goldenset/golden_ohb.csv", "골든셋/golden_ohb.csv"),
            ("eval/goldenset/golden_lse.csv", "골든셋/golden_lse.csv"),
        ],
    },
    {
        "dir": "05_발표자료",
        "원본": "`docs/발표자료/`",
        "요구": "(참고) 발표 자료",
        "핵심": "PetTriage_발표자료_3차.pptx",
        "한줄": "17장 — WHY · HOW 6장 · PROOF 4장 · 한계",
        "items": [
            ("docs/발표자료/PetTriage_발표자료_3차.pptx", "PetTriage_발표자료_3차.pptx"),
            ("docs/발표자료/PetTriage_발표자료_3차.pdf", "PetTriage_발표자료_3차.pdf"),
        ],
    },
    {
        "dir": "06_부속-설계기록",
        "원본": "`README.md` · `docs/`",
        "요구": "(참고) 기획 · 설계 원칙 · 결정 기록 · 역할 분담",
        "핵심": "06_설계결정기록.md  (86건 · D-01~D-92)",
        "한줄": "결정을 회의록으로 뭉뚱그리지 않고 그 시점에 적었다 — 팀원 기여는 그 자리에 표시",
        "items": [
            "README.md",
            "docs/00_기획-요구사항분석.md",
            "docs/03_모델-멀티태스크학습.md",
            "docs/03a_파인튜닝-구현기획.md",
            "docs/05_설계원칙-코드와LLM의분업.md",
            "docs/06_설계결정기록.md",
            "docs/07_작업배분.md",
            "docs/09_새환경-준비.md",
            "docs/도메인검토-반려동물-헬스케어.md",
            "docs/현재상태_2026-08-03.md",
            "docs/08_코드검토",
        ],
    },
]

# ──────────────────────────────────────────────────────────────
# ③ 구현 코드 — 복사하지 않고 지도를 그린다
# ──────────────────────────────────────────────────────────────

CODE_MAP = """# ③ 개발된 소프트웨어 — RAG 기반 LLM과 벡터 DB 연동 구현 코드

> ⚠️ **이 폴더에는 코드 사본이 없다.** 저장소 안에 코드가 두 벌 있으면 채점자가
> 어느 쪽을 읽어야 하는지 알 수 없고, 한쪽만 고쳐지는 순간 저장소가 거짓말을
> 시작한다. **원본은 `src/` 다.** 아래는 그리로 가는 지도다.

## 한 줄 요약

`POST /api/ask` 한 번에 **LangGraph 노드 18개**가 돈다. 벡터 검색은 **Chroma**(888청크 ·
`BAAI/bge-m3`), LLM 호출은 **LangChain `ChatOpenAI`**, 등급 계산은 **코드**가 한다.

## 필수 요건이 어느 파일에 있나

| 요건 | 파일 | 무엇을 보면 되나 |
|---|---|---|
| **LangChain 연동** | `src/pettriage/models/serving/client.py` | `LangChainClient` — `ChatOpenAI` 를 같은 `LLMClient` 프로토콜 뒤에 둔다 (D-71) |
| | `src/pettriage/models/serving/factory.py` | 설정 한 줄로 구현을 갈아끼운다 |
| | `tests/test_langchain_wiring.py` | 배선이 실제로 도는지 고정하는 테스트 |
| **벡터 DB** | `src/pettriage/retrieval/store.py` | Chroma 어댑터 · `to_chroma_where()` 필터 번역 |
| | `src/pettriage/retrieval/embedder.py` | `BAAI/bge-m3` 임베딩 |
| | `scripts/build_index.py` | 사실 표 → 청크 → Chroma 적재 |
| **RAG 파이프라인** | `src/pettriage/graph/build.py` | **`StateGraph` 조립 — 노드 18개와 조건부 엣지** |
| | `src/pettriage/graph/nodes/` | 노드 구현 7파일 (분류·슬롯·검색·계산·생성·판정·검증) |
| | `src/pettriage/graph/state.py` | 노드가 상태에 무엇을 쓸 수 있는지 스키마가 강제한다 |
| **안전 장치** | `src/pettriage/triage/gate.py` | 🔒 `apply_gate` — 하향 금지 · 정량 상향 차단 · 조건 없는 MONITOR 금지 |
| | `src/pettriage/triage/levels.py` | 4단계 정의와 **코퍼스 근거**(`EVIDENCE`) |
| | `src/pettriage/triage/basis.py` | 판정 근거를 `level` 과 같은 급의 필드로 낸다 (D-81) |
| | `src/pettriage/safety/contacts.py` | 마지막 관문 — 연락처 스크러빙 (D-47) |
| **수치 계산 (비-RAG)** | `src/pettriage/compute/rules.py` | 규칙 표 조회 — **수치는 벡터 검색으로 찾지 않는다** (D-16) |
| | `src/pettriage/compute/tables/` | 정량임계치 · 정성등급 · 물질어휘 (전부 **생성물**) |
| **API · DB** | `src/pettriage/app/main.py` · `routes/` | FastAPI 엔드포인트 14개 |
| | `src/pettriage/app/contracts.py` | **계약의 단일 출처** — 안전 불변식을 스키마로 강제 |
| | `src/pettriage/app/models.py` · `database.py` | MySQL · SQLAlchemy |
| | `src/pettriage/app/web/` | 화면 6장 (정적 HTML) |
| **전처리** | `src/pettriage/ingest/templates.py` · `verbalize.py` | 사실 표 → 문장. **LLM 이 아니라 코드가 만든다** (D-38) |
| **파인튜닝** | `src/pettriage/models/training/qlora.py` | Qwen3-4B QLoRA (담당 이서은) |
| **테스트** | `tests/` 27파일 | `pytest` **585 passed** |

## 돌려보기

```bash
make install          # pip install -e '.[api,rag,ingest,db,dev]' -c constraints.txt
make doctor           # 무엇이 빠졌는지 한 줄씩 짚어 준다
make index            # 사실 표 → 청크 → Chroma 적재
make serve            # http://127.0.0.1:8000
make test             # 안전 장치 회귀 테스트 585건
```

## 결과 재현

```bash
git checkout freeze-0803                       # 측정을 동결한 커밋
PETTRIAGE_PROFILE=eval python eval/harness/run_eval.py --arm none
```

`--arm none` 은 **API 키 없이** 돈다 — 기준선(통과 45.0%)을 누구나 다시 잴 수 있다.
`--arm A` 는 대형 LLM, `--arm A-LC` 는 같은 모델을 LangChain 으로 부른다.
"""


# ──────────────────────────────────────────────────────────────


def _force_rm(func, path, _exc):
    """Windows 에서 읽기 전용 속성이 붙은 파일은 `rmtree` 가 거부한다.

    `shutil.copy2` 가 원본의 권한을 그대로 복사하므로, 원본이 읽기 전용이면
    사본도 그렇게 되고 **다음 실행에서 폴더를 못 지운다.** 속성을 떼고 다시 시도한다.
    """
    import os
    import stat

    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:  # 그래도 안 되면 무엇이 잡고 있는지 알려 준다
        raise RuntimeError(
            f"{path} 를 지울 수 없다 — 탐색기·편집기에서 이 폴더를 열어 두었는지 확인해라."
        ) from e


def rmtree(path: Path) -> None:
    shutil.rmtree(path, onerror=_force_rm)


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        rev = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        return rev + (" (커밋되지 않은 변경 있음)" if dirty else "")
    except Exception:
        return "알 수 없음 (git 저장소가 아님)"


def human(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / 1024 / 1024:.1f} MB"
    if n >= 1024:
        return f"{n / 1024:.0f} KB"
    return f"{n} B"


def copy_group(group: dict, dry: bool) -> tuple[list[tuple[str, int]], list[str]]:
    """한 묶음을 복사한다. (복사된 것, 없는 것) 을 돌려준다."""
    copied: list[tuple[str, int]] = []
    missing: list[str] = []
    dest_root = OUT / group["dir"]

    for item in group["items"]:
        rel, sub = item if isinstance(item, tuple) else (item, Path(item).name)
        src = ROOT / rel
        dst = dest_root / sub

        if not src.exists():
            missing.append(rel)
            continue

        if dry:
            size = (
                sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
                if src.is_dir()
                else src.stat().st_size
            )
            copied.append((sub, size))
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
            size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
        else:
            shutil.copy2(src, dst)
            size = dst.stat().st_size
        copied.append((sub, size))

    return copied, missing


def write_readme(results: list[tuple[dict, list, list]]) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    A = lines.append

    A("# SKN_3rd_5Team_REPORT — 필수 산출물 모음")
    A("")
    A("**SKN 3차 단위 프로젝트 · 팀 save the pet** — PetTriage")
    A("")
    A("| | |")
    A("|---|---|")
    A("| 팀장 · 이 폴더 정리 | **오한빈** |")
    A("| 팀원 | 이근준 · 권소라 · 이서은 |")
    A("")
    A("## 📌 이 폴더가 무엇인가")
    A("")
    A("**여기 있는 문서는 전부 저장소의 다른 폴더에 이미 존재합니다.**")
    A("원본은 `docs/` · `data/` · `eval/` 에 흩어져 있고, 이 폴더는 그것을 옮긴 것이 아니라")
    A("**복사한 것**입니다.")
    A("")
    A("> **제출물을 받으시는 분이 필수 산출물 넷을 한자리에서 찾으실 수 있도록**")
    A("> `scripts/make_submission.py` 가 모아 둔 폴더입니다. 저장소를 돌아다니지 않으셔도")
    A("> 아래 대조표만 보시면 네 항목이 각각 어느 파일인지 바로 확인하실 수 있습니다.")
    A("")
    A("각 문서의 저장소 원본 위치는 항목마다 함께 적어 두었습니다.")
    A("")
    A("<details><summary>팀 내부용 — 이 폴더를 고치려면</summary>")
    A("")
    A("이 폴더는 **생성물이다. 손으로 고치지 않는다.**")
    A("안에 든 것은 전부 사본이라, 사본을 고치면 어느 쪽이 진짜인지 알 수 없게 된다 —")
    A("이 저장소가 데이터에 적용한 것과 같은 원칙이다 (D-22 · 단일 출처).")
    A("무엇이 바뀌었으면 **원본을 고치고 다시 만든다**:")
    A("")
    A("```bash")
    A("python scripts/make_submission.py")
    A("```")
    A("")
    A("</details>")
    A("")
    A(f"- 생성 시각 · `{now}`")
    A(f"- 저장소 커밋 · `{git_head()}`")
    A("- 측정 동결 커밋 · `freeze-0803`")
    A("")
    A("---")
    A("")
    A("## 필수 산출물 넷 — 어디를 보면 되나")
    A("")
    A("| 요구 항목 | 이 폴더 | 먼저 볼 파일 | 저장소 원본 |")
    A("|---|---|---|---|")
    for g, _copied, _missing in results:
        if g["dir"].startswith(("05", "06")):
            continue
        A(f"| **{g['요구']}** | `{g['dir']}/` | `{g['핵심']}` | {g['원본']} |")
    A("")
    A("참고 자료는 `05_발표자료/` 와 `06_부속-설계기록/` 에 있습니다.")
    A("")
    A("---")
    A("")

    for g, copied, missing in results:
        A(f"## {g['dir']}")
        A("")
        A(f"**{g['요구']}**")
        A("")
        A(f"> {g['한줄']}")
        A("")
        A(f"원본 위치 · {g['원본']}")
        A("")
        if copied:
            A("| 파일 | 크기 |")
            A("|---|---|")
            for name, size in copied:
                A(f"| `{name}` | {human(size)} |")
            A("")
        if g["dir"].startswith("03"):
            A(
                "코드 사본 대신 **경로 지도**를 두었습니다 — `03_구현-코드/README.md` 를 보시면 됩니다."
            )
            A("")
        if missing:
            A("🔴 **저장소에 없어 담지 못한 것**")
            A("")
            for m in missing:
                A(f"- `{m}`")
            A("")

    A("---")
    A("")
    A("## 이 프로젝트가 주장하는 것 한 줄")
    A("")
    A("**안전은 나중에 거르는 것이 아니라 미리 까는 것이다.**")
    A("일반적인 RAG 는 LLM 이 만든 답을 검증기가 걸러낸다 — 검증기가 뚫리면 방어가 없다.")
    A("이 시스템은 코드가 물질별 최소 등급을 먼저 깔고 LLM 은 그 위로만 움직인다.")
    A(
        "근거 검증이 300건에서 한 번도 조치를 실행하지 않은 상태에서도 **중대 과소평가는 다섯 판 전부 0건**이었다."
    )
    A("")

    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="복사하지 않고 목록만 본다")
    ap.add_argument("--zip", action="store_true", help="제출_<커밋>.zip 도 만든다")
    args = ap.parse_args()

    # 이름을 바꾼 뒤 남은 옛 폴더를 먼저 치운다 — 두 벌이 남으면 어느 쪽이 최신인지 모른다.
    if not args.dry_run:
        for old_name in LEGACY_NAMES:
            old = ROOT / old_name
            if old.is_dir() and (old / MARKER).exists():
                rmtree(old)
                print(f"· 옛 이름 폴더를 치웠다 — {old_name}/")
            elif old.is_dir():
                print(
                    f"⚠️ {old_name}/ 가 남아 있는데 생성물 표시가 없어 두었다. 확인 후 직접 지워라."
                )

    if OUT.exists() and not args.dry_run:
        if not (OUT / MARKER).exists():
            print(f"🔴 {OUT} 가 이미 있는데 생성물 표시({MARKER})가 없다.")
            print("   사람이 만든 폴더로 보고 지우지 않는다. 확인 후 직접 옮기거나 지워라.")
            return 1
        rmtree(OUT)

    if not args.dry_run:
        OUT.mkdir(parents=True)
        (OUT / MARKER).write_text(
            "이 폴더는 scripts/make_submission.py 가 만든다. 손으로 고치지 않는다.\n",
            encoding="utf-8",
        )

    results = []
    total = 0
    missing_total = 0
    for g in GROUPS:
        copied, missing = copy_group(g, args.dry_run)
        results.append((g, copied, missing))
        total += sum(s for _, s in copied)
        missing_total += len(missing)
        mark = "·" if not missing else "🔴"
        print(
            f"{mark} {g['dir']:<22} {len(copied):>2}개  {human(sum(s for _, s in copied)):>8}"
            + (f"   없음 {len(missing)}건" if missing else "")
        )
        for m in missing:
            print(f"     🔴 없음 — {m}")

    if args.dry_run:
        print(f"\n(--dry-run) 합계 {human(total)} · 없는 파일 {missing_total}건")
        return 0

    (OUT / GROUPS[2]["dir"]).mkdir(parents=True, exist_ok=True)
    (OUT / GROUPS[2]["dir"] / "README.md").write_text(CODE_MAP, encoding="utf-8")
    write_readme(results)

    print(
        f"\n✅ {OUT.relative_to(ROOT)}/  합계 {human(total)}"
        + (f" · 🔴 없는 파일 {missing_total}건 (README 에 적었다)" if missing_total else "")
    )

    if args.zip:
        rev = git_head().split()[0]
        name = ROOT / f"SKN_3rd_5Team_REPORT_{rev}"
        shutil.make_archive(str(name), "zip", root_dir=OUT)
        print(f"✅ {name.name}.zip ({human((ROOT / (name.name + '.zip')).stat().st_size)})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
