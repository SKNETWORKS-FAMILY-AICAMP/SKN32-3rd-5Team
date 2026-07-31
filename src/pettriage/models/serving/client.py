"""LLM 클라이언트 — 프로토콜 + 3구현.

설계 근거: docs/00 §5 · docs/06 D-04 · D-21 · D-42

    **대형 LLM 폴백 경로를 구조로 유지한다.** D-21이 정한 대로
    대형 LLM 기준 RAG를 먼저 완성하고 sLLM으로 교체하므로,
    두 구현이 같은 프로토콜 뒤에 있어야 교체가 설정 한 줄로 끝난다.

    04의 비교군(A: 대형 LLM / C: 파인튜닝 sLLM / D: 베이스 sLLM)이
    이 프로토콜의 구현 3종에 그대로 대응한다.

무거운 임포트는 함수 안에서 한다 — GPU 없이도 이 모듈을 읽을 수 있어야 한다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..tasks import Task


@runtime_checkable
class LLMClient(Protocol):
    """태스크 1건을 수행한다. 그래프 노드는 이 프로토콜만 안다."""

    name: str

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str: ...


class LocalQwenClient:
    """Qwen3-4B (+ LoRA 어댑터). 04 비교군 C·D.

    `adapter_path` 가 None이면 베이스 모델이므로 **비교군 D**가 된다.
    같은 클래스로 두 비교군을 돌릴 수 있어야 조건이 동일해진다.
    """

    def __init__(self, base_id: str, adapter_path: str | None = None, revision: str | None = None):
        self.name = f"qwen:{base_id}" + (f"+{adapter_path}" if adapter_path else ":base")
        self._base_id = base_id
        self._adapter = adapter_path
        self._revision = revision
        self._model = None
        self._tok = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self._base_id, revision=self._revision)
        model = AutoModelForCausalLM.from_pretrained(
            self._base_id,
            revision=self._revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if self._adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self._adapter)
        self._model = model.eval()

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str:
        from ..prompts import build_messages

        self._ensure()
        assert self._tok is not None and self._model is not None
        text = self._tok.apply_chat_template(
            build_messages(task, user_input), tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # 04 §8 — 평가 재현성. 샘플링을 쓰지 않는다
        )
        return self._tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)


class APIClient:
    """대형 LLM (04 비교군 A · 폴백 경로).

    ⚠️ D-36 — 여기로 나가는 입력은 **개인정보 필터를 통과한 것만**이어야 한다.
    필터는 호출부가 아니라 `privacy/` 가 강제한다.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.name = f"api:{model}"
        self._model = model

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str:
        from openai import OpenAI

        from ...config import get_secrets
        from ..prompts import build_messages

        key = get_secrets().openai_api_key
        client = OpenAI(api_key=key.get_secret_value() if key else None)
        resp = client.chat.completions.create(
            model=self._model,
            messages=build_messages(task, user_input),  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content or ""


class EchoClient:
    """테스트용. 모델 없이 그래프·계약을 돌린다."""

    name = "echo"

    def __init__(self, responses: dict[Task, str] | None = None):
        self._responses = responses or {}

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str:
        return self._responses.get(task, "")
