"""LLM 폴백 기록 — **어느 태스크가 모델 없이 처리됐나** (05 §6 · D-22).

    reset_llm_fallbacks()          # 요청 하나의 경계에서 (엔진이 부른다)
    note_fallback(Task.CLASSIFY)   # 모델이 없거나 실패했을 때 (노드가 부른다)
    current()                      # 정렬된 사본 (엔진이 응답에 싣는다)

## 왜 별도 모듈인가

    2026-08-02 까지 이 집합은 `nodes/generate.py` 안에 있었고, **거기 있는 노드만**
    기록했다. ①분류(`classify.py`)와 ②슬롯(`slots.py`)은 `get_client()` 가 `None`
    이어도 조용히 `None` 을 돌려주고 끝이었다.

    그래서 04 §3 비교군 A 를 재도 **"5태스크 중 몇 개가 실제로 LLM 을 탔나"** 를
    셀 수 없었다. 성적이 나쁘게 나오면 *모델이 못한 것*인지 *모델이 안 불린 것*인지
    구분이 안 된다 — 이서은 팀원이 잡은 D-73(라벨 누락으로 LLM 이 폴백보다 나빴다)이
    정확히 그 구분이 안 돼 오래 안 보였던 문제다.

    기록하는 자리가 특정 노드 파일 안에 있으면 **다른 노드는 기록하지 않는 것이 기본값**이
    된다. 노드 밖으로 꺼내 다섯 태스크가 같은 문을 쓰게 한다 (D-40 — 못 어기게 둔다).

## 이 집합은 프로세스 전역이다

    서버는 요청을 이어 처리하므로 그대로 읽으면 앞 요청의 기록이 섞인다.
    **비우는 것도 읽는 것도 요청 경계에서 한 번씩**, 그 경계를 아는 `GraphEngine` 이 한다
    (`graph/engine.py::_run_pipeline`). 노드는 더하기만 한다.

    ⚠️ 스레드 안전하지 않다. 지금 하네스·테스트·개발 서버는 요청을 겹쳐 돌리지 않는다.
       동시 요청을 받게 되면 이 집합이 아니라 **상태 채널**로 옮겨야 한다.
"""

from __future__ import annotations

from ..models.tasks import Task

#: 이번 요청에서 **폴백으로 처리된 태스크 이름.**
#:
#: 04 §8 이 *"조용히 스텁으로 내려가면 지표가 스텁으로 산출된다. 그 지표는 오염된
#: 것이므로 기본은 실패다"* 라고 한 것을 LLM 폴백에도 적용한다. **끄지 않고 표시한다.**
LLM_FALLBACKS: set[str] = set()

#: 5태스크 밖의 호출(초안·트리아지 판정)이 실패했을 때 쓰는 이름.
#: 파인튜닝 대상이 아니므로 태스크 이름을 빌리지 않는다 (05 §4).
RAW = "(raw)"


def note_fallback(task: Task | str) -> None:
    """이 태스크는 **모델 없이** 처리됐다고 남긴다."""
    LLM_FALLBACKS.add(task.value if isinstance(task, Task) else task)


def reset_llm_fallbacks() -> None:
    """측정·요청 시작 전에 비운다. **엔진과 하네스만 부른다.**"""
    LLM_FALLBACKS.clear()


def current() -> list[str]:
    """정렬된 사본. 집합을 그대로 넘기면 받는 쪽이 나중에 바뀐 것을 본다."""
    return sorted(LLM_FALLBACKS)
