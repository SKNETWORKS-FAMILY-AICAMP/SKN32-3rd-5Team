"""서빙 클라이언트 팩토리 — **어느 모델로 답하는가는 설정이 정한다** (D-40 · D-65).

설계 근거: 05 §4 · 04 §3 · D-21 · D-42

    `client.py` 머리말은 처음부터 이렇게 적혀 있었다 —

        "두 구현이 같은 프로토콜 뒤에 있어야 교체가 **설정 한 줄로** 끝난다"

    **그 한 줄이 없었다.** 2026-08-02 점검에서 노드 4곳이 `APIClient()` 를 직접
    만들고 있었고, `LocalQwenClient` 는 소스 어디에서도 생성되지 않았다.
    `configs/*.yaml` 의 `model.base_id`·`revision`·`adapter_path` 는 학습 경로만
    읽었다. 그래서 **04 §3 비교표의 C·D 열을 채울 방법 자체가 없었다.**

    D-64 와 같은 모양이다 — 문서에 적힌 결정이 코드에서 강제되지 않았다.

## 비교군과의 대응

    provider    무엇                                        04 §3
    ─────────────────────────────────────────────────────────────
    none        모델 없음. 5태스크 전부 폴백               (기준선)
    api         대형 LLM (`model.api_model`) — openai SDK 직접   A
    langchain   같은 모델을 **LangChain 으로** (D-71)          A(LC)
    qwen        Qwen3-4B 베이스 (`adapter_path` 없음)        D
    qwen        Qwen3-4B + LoRA (`adapter_path` 있음)        C
    echo        고정 응답 (테스트)                            —

    **`none` 을 명시적인 값으로 둔다.** 예전에는 *"키가 없으면 폴백"* 이라
    모델 없이 돈 것이 **설정이 아니라 사고**였고, 리포트에 그 조건을 적을
    이름이 없었다. 이름이 있어야 측정 조건으로 기록된다 (04 §8).

## 상주

    `get_client` 에 `lru_cache` 가 붙어 있다. Qwen 4B 는 로드가 비싸고,
    노드마다 새로 만들면 **질의 하나에 모델을 6번 올린다** (D-53).
    설정을 바꿔 다시 만들려면 `reset_client()` 를 부른다.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from .client import LLMClient

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_client() -> LLMClient | None:
    """설정이 가리키는 클라이언트. **없으면 `None`** — 부르는 쪽이 폴백한다.

    ⚠️ **예외를 던지지 않는다.** 05 §6 이 정한 실패 방식은 *①분류는 폴백 + 로그 ·
    ②슬롯은 되묻기* 다. 모델이 없다고 그래프가 죽으면 그 경로를 우회한다.

    `None` 이 되는 경우는 둘이다 — `provider="none"`, 그리고 `provider="api"`
    인데 키가 없을 때. **후자는 로그로 남긴다** — 키를 넣었다고 생각하는 사람이
    폴백 성적을 LLM 성적으로 읽는 사고를 막는다.
    """
    from ...config import get_config, get_secrets

    m = get_config().model

    if m.provider == "none":
        return None

    if m.provider == "echo":
        from .client import EchoClient

        return EchoClient()

    if m.provider in ("api", "langchain"):
        if not get_secrets().openai_api_key:
            # **실제 provider 를 찍는다.** `api` 로 고정돼 있어서 `langchain` 으로
            # 돌렸는데 로그는 `api` 라고 말했다 — 로그가 거짓이면 진단이 어긋난다.
            log.warning(
                "model.provider=%s 인데 OPENAI_API_KEY 가 없다 — "
                "5태스크가 전부 폴백으로 돈다. 의도한 것이면 provider=none 으로 두면 "
                "그 사실이 설정에 남는다 (04 §8).",
                m.provider,
            )
            return None
        if m.provider == "langchain":
            from .client import LangChainClient

            return LangChainClient(model=m.api_model, base_url=m.api_base_url)
        from .client import APIClient

        return APIClient(model=m.api_model, base_url=m.api_base_url)

    if m.provider == "qwen":
        from .client import LocalQwenClient

        return LocalQwenClient(
            base_id=m.base_id,
            adapter_path=m.adapter_path,
            revision=m.revision,  # ← 이 핀이 서빙에 걸리는 것은 여기가 처음이다
            dtype=m.dtype,
            load_in_4bit=m.load_in_4bit,
        )

    # Literal 로 막혀 있어 도달할 수 없다. 도달했다면 설정 검증이 뚫린 것이다.
    log.error("알 수 없는 model.provider: %r — 모델 없이 진행한다", m.provider)
    return None


def reset_client() -> None:
    """캐시를 비운다. 설정을 바꿔 가며 비교군을 도는 하네스가 부른다."""
    get_client.cache_clear()


def client_name() -> str:
    """리포트에 박을 이름. **무엇으로 잰 건지 모르는 숫자를 남기지 않는다** (04 §8)."""
    c = get_client()
    return c.name if c is not None else "none(폴백)"
