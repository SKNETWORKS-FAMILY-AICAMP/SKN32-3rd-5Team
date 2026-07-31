"""테스트 공통 픽스처.

이 프로젝트에는 프로세스 전역 상태가 셋 있다 — 설정 캐시(`lru_cache`),
엔진·세션·기록 저장소(`deps` 모듈 전역), 그리고 `main.app` 임포트 시점의 앱 인스턴스.

전역을 그대로 두면 **앞 테스트가 뒤 테스트를 오염시키고**, 그런 통과는
순서에 의존하는 가짜다. 그래서 매 테스트마다 초기화한다.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pettriage import config as config_mod
from pettriage.app import deps
from pettriage.app.main import create_app


@pytest.fixture(autouse=True)
def _isolate_global_state(monkeypatch: pytest.MonkeyPatch):
    """설정 캐시와 전역 저장소를 테스트마다 비운다."""
    monkeypatch.delenv("PETTRIAGE_PROFILE", raising=False)
    for key in list(__import__("os").environ):
        if key.startswith("PETTRIAGE__"):
            monkeypatch.delenv(key, raising=False)
    config_mod.reset_caches()
    deps.reset_state()
    yield
    config_mod.reset_caches()
    deps.reset_state()


@pytest.fixture
def client() -> TestClient:
    """앱을 매번 새로 만든다 — 미들웨어·설정이 테스트마다 독립이어야 한다."""
    return TestClient(create_app())
