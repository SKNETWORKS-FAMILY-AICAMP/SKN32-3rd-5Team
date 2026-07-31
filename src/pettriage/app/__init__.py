"""배달 계층 — FastAPI + 정적 프론트.

`main.create_app()` 이 진입점이다. 계약은 `contracts.py`,
엔진 교체 지점은 `deps.py` 한 곳뿐이다.
"""

__all__ = ["create_app"]


def __getattr__(name: str):
    """지연 임포트 — fastapi 미설치 환경에서도 패키지 임포트가 깨지지 않게."""
    if name == "create_app":
        from .main import create_app

        return create_app
    raise AttributeError(name)
