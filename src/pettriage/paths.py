"""프로젝트 루트 탐색 — 설치 형태에 따라 달라지는 경로를 한 곳에서 푼다.

설계 근거: docs/04_테스트-평가계획.md §8 재현성

    ``Path(__file__).parents[N]`` 은 소스 트리에서만 맞는다.
    비편집 설치(site-packages)에서는 엉뚱한 곳을 가리키고, 그러면
    ``configs/`` 를 못 찾아 **조용히 기본값으로 되돌아간다.**

    평가 프로파일이 무시된 채 지표가 산출되면 그 지표는 오염된 것이다.
    그래서 여기서는 **못 찾으면 조용히 넘어가지 않는다** — 부르는 쪽이
    경고하거나 실패하도록 `None` 을 명확히 돌려준다.

우선순위
    1. ``PETTRIAGE_ROOT`` 환경변수 (컨테이너·배포에서 명시)
    2. 모듈 위치에서 위로 올라가며 표식(``configs/`` · ``pyproject.toml``) 탐색
    3. 현재 작업 디렉터리에서 같은 탐색
"""

from __future__ import annotations

import os
from pathlib import Path

#: 이 중 하나라도 있으면 프로젝트 루트로 본다.
_MARKERS = ("configs", "pyproject.toml")


def _looks_like_root(p: Path) -> bool:
    return any((p / m).exists() for m in _MARKERS)


def _search_upward(start: Path) -> Path | None:
    for cand in (start, *start.parents):
        if _looks_like_root(cand):
            return cand
    return None


def find_root() -> Path | None:
    """프로젝트 루트. 찾지 못하면 ``None`` — 조용히 추측하지 않는다."""
    env = os.getenv("PETTRIAGE_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None

    here = Path(__file__).resolve().parent
    return _search_upward(here) or _search_upward(Path.cwd().resolve())


def config_dir() -> Path | None:
    """``configs/`` 디렉터리. ``PETTRIAGE_CONFIG_DIR`` 로 직접 지정할 수 있다."""
    env = os.getenv("PETTRIAGE_CONFIG_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        return p if p.is_dir() else None
    root = find_root()
    if root is None:
        return None
    d = root / "configs"
    return d if d.is_dir() else None


def web_dir() -> Path | None:
    """데모 프론트 디렉터리.

    **패키지 안(`app/web/`)을 먼저 본다.** 프론트는 배달 계층의 일부이므로
    패키지와 함께 설치되어야 한다 — 저장소 루트에 두면 `pip install` 후
    화면이 사라진다.
    """
    packaged = Path(__file__).resolve().parent / "app" / "web"
    if packaged.is_dir():
        return packaged
    root = find_root()
    if root is None:
        return None
    d = root / "web"  # 구버전 배치 호환
    return d if d.is_dir() else None


def data_dir() -> Path:
    """자료 디렉터리. 루트를 못 찾으면 현재 디렉터리 기준으로 둔다."""
    root = find_root()
    return (root or Path.cwd()) / "data"
