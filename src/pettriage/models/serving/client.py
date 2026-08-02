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

import logging
from typing import Protocol, runtime_checkable

from ..tasks import Task

log = logging.getLogger(__name__)


@runtime_checkable
class LLMClient(Protocol):
    """태스크 1건을 수행한다. 그래프 노드는 이 프로토콜만 안다."""

    name: str

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str: ...

    def run_raw(self, system: str, user_input: str, *, max_tokens: int = 512) -> str:
        """**5태스크 밖**의 호출 (답변 생성 · 트리아지 판정).

        ⚠️ 프로토콜에 이것이 없어서 `LocalQwenClient` 만 구현을 빠뜨렸고,
        `generate.py` 가 부르는 순간 `AttributeError` 가 났다 — 그런데
        아무도 그 클라이언트를 만들지 않아 **드러나지 않았다** (2026-08-02).
        프로토콜이 요구하지 않으면 구현이 빠져도 아무것도 알려주지 않는다.
        """
        ...


class LocalQwenClient:
    """Qwen3-4B (+ LoRA 어댑터). 04 비교군 C·D.

    `adapter_path` 가 None이면 베이스 모델이므로 **비교군 D**가 된다.
    같은 클래스로 두 비교군을 돌릴 수 있어야 조건이 동일해진다.
    """

    def __init__(
        self,
        base_id: str,
        adapter_path: str | None = None,
        revision: str | None = None,
        *,
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
    ):
        self.name = f"qwen:{base_id}" + (f"+{adapter_path}" if adapter_path else ":base")
        self._base_id = base_id
        self._adapter = adapter_path
        self._revision = revision
        self._dtype = dtype
        self._4bit = load_in_4bit
        self._model = None
        self._tok = None

    def _ensure(self) -> None:
        """가중치를 올린다. **환경이 못 해 주는 것은 폴백하고 로그를 남긴다** (05 §6).

        ⚠️ 예전에는 `torch_dtype=bfloat16, device_map="auto"` 가 **박혀 있었다.**
        `configs` 의 `dtype`·`load_in_4bit` 를 안 읽었고, 그 차이가 실질적이다 —

            bf16   VRAM 약 8~9GB      ← 코드가 하던 것
            4bit   VRAM 약 3~4GB      ← 설정이 요구하던 것

        노트북 GPU 에서는 이 차이가 **되냐 안 되냐**를 가른다. D-65 와 같은 사각지대였다.

        폴백은 **끄지 않고 표시한다** — 4bit 로 재려던 실험이 조용히 bf16 으로
        돌면 04 §8 재현성이 무너진다.
        """
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        cuda = torch.cuda.is_available()
        kw: dict = {"revision": self._revision}

        # ① dtype — **CPU 에서는 bfloat16 을 쓰지 않는다.** 지원은 되지만 느리다.
        want = {"bfloat16": torch.bfloat16, "float16": torch.float16, "auto": "auto"}.get(
            self._dtype, torch.bfloat16
        )
        kw["torch_dtype"] = want if cuda else torch.float32
        if not cuda:
            log.warning("GPU 가 없다 — float32/CPU 로 올린다. 4B 생성은 매우 느리다.")

        # ② 4bit — bitsandbytes 가 있고 GPU 가 있을 때만.
        if self._4bit and cuda:
            try:
                from transformers import BitsAndBytesConfig

                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                )
                kw.pop("torch_dtype", None)
            except Exception as e:  # noqa: BLE001
                log.warning("load_in_4bit=true 인데 4bit 를 못 쓴다 (%s) — %s 로 올린다.", e, want)
        elif self._4bit:
            log.warning("load_in_4bit=true 인데 GPU 가 없다 — 무시한다.")

        # ③ device_map="auto" 는 accelerate 를 요구한다. 없으면 단순 로드.
        try:
            import accelerate  # noqa: F401

            kw["device_map"] = "auto"
        except ImportError:
            log.warning(
                "accelerate 가 없다 — device_map='auto' 없이 올린다. `pip install accelerate`"
            )

        self._tok = AutoTokenizer.from_pretrained(self._base_id, revision=self._revision)
        model = AutoModelForCausalLM.from_pretrained(self._base_id, **kw)
        if "device_map" not in kw:
            model = model.to("cuda" if cuda else "cpu")
        if self._adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self._adapter)
        self._model = model.eval()
        log.info("qwen 로드 완료 — %s", {k: str(v) for k, v in kw.items() if k != "revision"})

    def _generate(self, messages: list[dict[str, str]], max_tokens: int) -> str:
        """메시지 → 생성. `run` 과 `run_raw` 가 **같은 경로**를 쓴다.

        갈라 두면 한쪽만 `do_sample=False` 를 빠뜨리는 식으로 조용히 어긋난다.

        `enable_thinking=False` — Qwen3는 기본적으로 답 앞에 `<think>...</think>`
        추론 블록을 길게 쓴다. 5태스크는 전부 짧고 구조화된 출력이 목표이고
        (05 §4 — 라벨 하나·JSON 하나), 학습 샘플의 target도 사고 과정 없이
        바로 답만 담고 있다(`prompts.build_sample`). 추론에서 생각 모드를 켜 두면
        `max_tokens`이 사고 과정에서 다 소진돼 진짜 답이 나오기 전에 잘리고,
        학습·추론 프롬프트가 어긋난다(이 모듈 머리말 "같은 문자열을 쓴다").
        """
        self._ensure()
        assert self._tok is not None and self._model is not None
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,  # 04 §8 — 평가 재현성. 샘플링을 쓰지 않는다
        )
        return self._tok.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str:
        from ..prompts import build_messages

        return self._generate(build_messages(task, user_input), max_tokens)

    def run_raw(self, system: str, user_input: str, *, max_tokens: int = 512) -> str:
        """5태스크 **밖**의 호출. 파인튜닝 태스크를 빌려 쓰지 않는다 (04 §3)."""
        return self._generate(
            [{"role": "system", "content": system}, {"role": "user", "content": user_input}],
            max_tokens,
        )


class APIClient:
    """대형 LLM (04 비교군 A · 폴백 경로).

    ⚠️ D-36 — 여기로 나가는 입력은 **개인정보 필터를 통과한 것만**이어야 한다.
    필터는 호출부가 아니라 `privacy/` 가 강제한다.
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None):
        # 이름에 엔드포인트를 넣는다 — 리포트에 `api:Qwen/Qwen3-4B` 로만 박히면
        # **어디서 서빙한 가중치인지 나중에 알 수 없다** (04 §8).
        self.name = f"api:{model}" + (f"@{base_url}" if base_url else "")
        self._model = model
        self._base_url = base_url

    def _client(self):
        from openai import OpenAI

        from ...config import get_secrets

        key = get_secrets().openai_api_key
        return OpenAI(
            api_key=key.get_secret_value() if key else None,
            base_url=self._base_url,  # None 이면 OpenAI 본가
        )

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str:
        from ..prompts import build_messages

        resp = self._client().chat.completions.create(
            model=self._model,
            messages=build_messages(task, user_input),  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0,
        )
        return resp.choices[0].message.content or ""

    def run_raw(self, system: str, user_input: str, *, max_tokens: int = 512) -> str:
        """**5태스크 밖**의 호출 (답변 생성 · 트리아지 판정).

        05 §4 가 LLM 에 맡긴 5태스크는 파인튜닝 대상이고 04 §3 이 지표를 잰다.
        그 밖의 호출이 태스크를 빌려 쓰면 **무엇을 잰 건지 모르게 된다.**
        그래서 시스템 프롬프트를 직접 받는 문을 따로 둔다.

        ⚠️ D-36 — 여기로 나가는 입력도 개인정보 필터를 통과한 것이어야 한다.
        """
        resp = self._client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ],
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

    def run_raw(self, system: str, user_input: str, *, max_tokens: int = 512) -> str:
        return self._responses.get(system, "")  # type: ignore[call-overload]
