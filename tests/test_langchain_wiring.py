"""**LangChain 이 실제로 쓰이는가** (D-71 · 필수 산출물).

    · LangChain을 활용하여 벡터데이터베이스와 LLM 연동

2026-08-02 점검에서 `langchain` 이 소스 어디에서도 **쓰이지 않는다**는 것이 드러났다.
쓰이는 것은 `langgraph`(오케스트레이션)뿐이었고, LLM 은 `openai` SDK 를,
벡터DB는 `chromadb` 를 직접 부르고 있었다. `config.py` 의 `langchain_api_key`
한 줄이 전부였다 — **이름만 있고 물건이 없었다** (D-64 와 같은 모양).

문서로 적어 두는 것으로는 다시 안 생긴다는 보장이 없다. 여기서 막는다.
"""

from __future__ import annotations

import pytest

#: ⚠️ `langchain-openai` 는 `[rag]` 에 있다. CI test 잡은 `.[api,dev]` 만 깐다.
#:
#: **모듈 최상단에서 임포트하면 그 잡이 통째로 죽는다** — 방금 D-70 으로 기록한
#: 바로 그 사고다(`chat_logger` → `sqlalchemy`). 검사 파일도 예외가 아니다.
#:
#: 대신 **건너뛴 것이 보이게** 둔다. `rag-deps` 잡이 `.[api,rag,ingest,dev]` 를 깔고
#: 임포트를 확인하므로, 거기에 이 파일을 돌리는 단계를 넣으면 실제로 검사된다.
#: 지금은 그 잡이 *"설치가 해결되는가"* 만 본다 — **건너뛴 검사는 통과가 아니다** (04 §8).
pytest.importorskip("langchain_openai", reason="[rag] 에만 있다 — rag-deps 잡에서 검사한다")


def test_langchain_client_is_importable_and_named():
    """`provider=langchain` 이 진짜 LangChain 객체를 만든다."""
    from langchain_core.language_models import BaseChatModel

    from pettriage.models.serving.client import LangChainClient

    c = LangChainClient(model="gpt-4o-mini")
    assert c.name.startswith("langchain:")
    # 키가 없어도 **구성은 된다** — 실제 호출만 키를 요구한다.
    chat = c._chat(max_tokens=16)  # noqa: SLF001
    assert isinstance(chat, BaseChatModel), type(chat)


def test_prompts_come_from_one_place():
    """**프롬프트의 단일 출처는 `models/prompts.py` 다** (D-22).

    LangChain 을 쓴다고 프롬프트가 갈라지면 04 §3 이 태스크별로 재는 지표가
    **무엇을 잰 건지 모르게 된다.** 두 클라이언트가 같은 문장을 보내야 한다.
    """
    from pettriage.models.prompts import build_messages
    from pettriage.models.serving.client import LangChainClient
    from pettriage.models.tasks import Task

    pairs = build_messages(Task.CLASSIFY, "강아지가 초콜릿을 먹었어요")
    msgs = LangChainClient._to_messages(pairs)  # noqa: SLF001

    assert len(msgs) == len(pairs)
    for src, got in zip(pairs, msgs, strict=True):
        assert got.content == src["content"], "프롬프트가 변형됐다"


def test_roles_map_to_langchain_message_types():
    """역할이 뒤바뀌면 시스템 지시가 사용자 발화로 나간다 — 조용히 품질만 떨어진다."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    from pettriage.models.serving.client import LangChainClient

    msgs = LangChainClient._to_messages(  # noqa: SLF001
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert [type(m) for m in msgs] == [SystemMessage, HumanMessage, AIMessage]


def test_langchain_arm_selects_the_langchain_client(monkeypatch: pytest.MonkeyPatch):
    """`--arm A-LC` 가 실제로 그 구현을 고른다 (arms.py)."""
    monkeypatch.setattr("os.environ", dict(__import__("os").environ))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    from pettriage.models.serving.arms import apply_arm
    from pettriage.models.serving.client import APIClient, LangChainClient
    from pettriage.models.serving.factory import get_client

    apply_arm("A-LC")
    assert isinstance(get_client(), LangChainClient)

    apply_arm("A")
    got = get_client()
    assert isinstance(got, APIClient) and not isinstance(got, LangChainClient)


def test_vector_store_is_deliberately_not_langchain():
    """**벡터DB는 chromadb 를 직접 쓴다** — 게을러서가 아니라 그래야 할 것이 있어서다.

    `filter_by_threshold`(D-46 실측 임계값)·`dedupe_by_substance`·
    `to_chroma_where`(D-56)·`publisher`/`locator` 메타(D-37)·`Hit.merged_sources`.
    **D-46 은 실측에서 나온 우리 도메인의 사실이고, 그런 판단은 추상 뒤에서 못 한다.**

    이 테스트는 금지가 아니라 **기록**이다 — 나중에 LangChain VectorStore 로
    옮긴다면 위 다섯을 어떻게 유지할지 먼저 답해야 한다는 표시다 (D-71).
    """
    from pettriage import retrieval

    assert hasattr(retrieval, "filter_by_threshold")
    assert hasattr(retrieval, "dedupe_by_substance")
    assert hasattr(retrieval, "ChromaStore")


def test_thinking_is_disabled_for_gemini_only():
    """**사고 모드를 끈다 — Gemini 에만** (2026-08-02 실측).

    안 끄면 사고 토큰이 `max_tokens` 를 먼저 다 쓰고 **본문 전에 잘린다.**
    ①분류(`max_tokens=16`)가 빈 문자열을 돌려주고, 코드는 그것을 허용목록 밖으로
    걸러 전부 `unknown` → 거절로 보냈다. Qwen3 로컬에서 만난 것과 같은 함정이다.

    ⚠️ **OpenAI 본가에 보내면 400 이다.** 추론 모델이 아닌 모델은 이 인자를 모른다.
    """
    from pettriage.models.serving.client import APIClient

    gem = "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert APIClient("gemini-3.5-flash", gem)._thinking_off() == {"reasoning_effort": "none"}
    assert APIClient("gpt-4o-mini")._thinking_off() == {}
    assert APIClient("qwen/qwen3-4b", "https://openrouter.ai/api/v1")._thinking_off() == {}


def test_both_clients_share_retry_and_thinking_rules():
    """A 와 A-LC 는 **연동 방식만** 달라야 한다 (D-71 · D-22).

    재시도 횟수나 사고 설정이 다르면 두 실행의 차이가 *"LangChain 때문"* 인지
    *"조건이 달라서"* 인지 알 수 없다 — 비교가 성립하지 않는다.
    """
    from pettriage.models.serving import client as m

    gem = "https://generativelanguage.googleapis.com/v1beta/openai/"
    lc = m.LangChainClient("gemini-3.5-flash", gem)._chat(max_tokens=16)
    assert lc.max_retries == m._MAX_RETRIES
    assert lc.reasoning_effort == "none"
