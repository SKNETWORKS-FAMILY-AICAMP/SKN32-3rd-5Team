"""`GraphEngine` — 그래프를 배달 계층에 물린다.

설계 근거: docs/06 D-40 · docs/02 §12.1

    `deps.get_engine()` 이 `configs/*.yaml` 의 `serve.engine` 을 보고 고른다.
    `graph` 로 두면 이 클래스가 물리고, 계약·프론트·테스트는 그대로다.

⚠️ **노드가 아직 비어 있다.** 지금 `serve.engine=graph` 로 띄우면
`EngineNotReady` 로 크게 실패한다 — 조용히 스텁으로 내려가면
평가 결과가 오염되기 때문이다 (04 §8).
"""

from __future__ import annotations

from ..app.contracts import AskRequest, AskResponse
from ..app.session import Session


class EngineNotReady(RuntimeError):
    """그래프 노드가 아직 구현되지 않았다.

    `pytest -m todo` 로 남은 일을 확인한다.
    """


class GraphEngine:
    """LangGraph 기반 질의 엔진 — **WS2 구현 대기.**

    완료 기준:
      1. `pytest -m todo` 가 전부 통과한다
      2. `PETTRIAGE__SERVE__ENGINE=graph` 로 띄워 `/api/ask` 가 세 상태를 모두 낸다
      3. `tests/test_api.py` 68건이 그대로 통과한다 — 계약은 바뀌지 않는다
    """

    name = "graph"

    def __init__(self) -> None:
        from .nodes import NODES_IMPLEMENTED

        # 생성 시점에 막는다 — 반쯤 구현된 그래프로 평가를 돌리면
        # 지표가 오염되고, 그 사실이 결과에 드러나지 않는다 (04 §8).
        if not NODES_IMPLEMENTED:
            raise EngineNotReady(
                "그래프 노드가 비어 있다. src/pettriage/graph/nodes/ 를 구현하고 "
                "nodes/__init__.py 의 NODES_IMPLEMENTED 를 True 로 바꿀 것. "
                "남은 일: pytest -m todo"
            )

    def ask(self, req: AskRequest, session: Session) -> AskResponse:
        raise NotImplementedError("WS2: 노드를 엮어 그래프를 실행한다 (02 §6)")
