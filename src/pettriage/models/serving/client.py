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

#: 429(속도 제한) 재시도 횟수. SDK 기본은 2회이고, 60건 × 6회에는 모자란다.
_MAX_RETRIES = 8
#: 한 호출의 상한(초). 사고 모델은 느릴 수 있다.
_TIMEOUT_S = 120.0

#: 마지막 호출 시각. `model.min_interval_ms` 를 지키는 데 쓴다.
#: 하네스는 단일 스레드라 모듈 변수로 충분하다 — 동시 호출이 생기면 락이 필요하다.
_last_call_at = 0.0


def _throttle() -> None:
    """`model.min_interval_ms` 만큼 **기다린다.**

    ⚠️ 저등급 API 에서 429 가 쏟아지면 재시도가 흡수하려 애쓰다 실행이 멈춘다.
    **막고 기다리는 편이 맞고 부딪히고 재시도하는 것보다 낫다** — 재시도는
    실패한 뒤의 대응이고, 이건 실패를 안 만드는 쪽이다.

    설정이 0이면 아무것도 하지 않는다. 기본이 0이므로 **명시적으로 켜야** 한다.
    """
    global _last_call_at
    import time

    try:
        from ...config import get_config

        gap = get_config().model.min_interval_ms / 1000.0
    except Exception:  # noqa: BLE001 — 설정을 못 읽어도 호출은 나가야 한다
        return
    if gap <= 0:
        return
    wait = gap - (time.monotonic() - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


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

        # ① dtype — 설정값을 따르되, **CPU 에서 못 쓰는 것만 바꾼다.**
        #
        # ⚠️ 처음에는 CPU 면 무조건 float32 로 올렸다. 그런데 4B 를 float32 로 올리면
        #    **RAM 약 16GB** 다 — 노트북에서는 스왑하거나 죽는다. bfloat16 은 CPU 에서
        #    느리지만 **8GB 로 절반**이고, 스왑보다는 느린 편이 낫다.
        #    float16 만 CPU 에서 연산이 제대로 안 돌아 float32 로 올린다.
        want = {"bfloat16": torch.bfloat16, "float16": torch.float16, "auto": "auto"}.get(
            self._dtype, torch.bfloat16
        )
        if not cuda and want is torch.float16:
            log.warning("CPU 에서 float16 은 연산이 안 된다 — float32 로 올린다 (RAM 2배).")
            want = torch.float32
        kw["torch_dtype"] = want
        if not cuda:
            log.warning(
                "GPU 가 없다 — CPU 로 올린다 (dtype=%s). 4B 생성은 매우 느리다. "
                "RAM 은 bfloat16 ≈8GB · float32 ≈16GB 다.",
                self._dtype,
            )

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
        """
        self._ensure()
        assert self._tok is not None and self._model is not None
        # ⚠️ **Qwen3 는 기본이 사고 모드(thinking)다.**
        #
        #   그냥 부르면 답 앞에 `<think> … </think>` 를 수백 토큰 쏟는다. 우리 태스크는
        #   ①분류가 `max_tokens=16`, ④검증이 짧은 라벨이라 **사고 토큰만 내고 잘린다** —
        #   라벨이 아예 안 나오고, 코드는 그것을 허용목록 밖으로 걸러 폴백한다.
        #   D-73 으로 프롬프트에 라벨을 실어도 **모델이 답을 시작하기 전에 끝난다.**
        #
        #   CPU 에서는 그 사고 토큰이 그대로 시간이다 — 느린 것의 절반이 이것이다.
        #   04 §8 재현성 관점에서도 사고 길이가 실행마다 달라지면 지연이 안 재진다.
        #
        #   `enable_thinking` 은 Qwen3 계열 템플릿에만 있다. 다른 모델에서도 돌아야 하므로
        #   **받아주는 경우에만** 넘긴다 — 없는 인자를 넘기면 TypeError 로 죽는다.
        kw = {"tokenize": False, "add_generation_prompt": True}
        try:
            text = self._tok.apply_chat_template(messages, enable_thinking=False, **kw)
        except TypeError:
            text = self._tok.apply_chat_template(messages, **kw)
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            # 04 §8 — 평가 재현성. 샘플링을 쓰지 않는다.
            do_sample=False,
            # ⚠️ **모델이 들고 온 샘플링 값을 명시적으로 지운다.**
            #    Qwen3 의 `generation_config.json` 은 temperature=0.6 · top_p=0.95 ·
            #    top_k=20 을 담고 있다. `do_sample=False` 면 안 쓰이지만 **남아 있다.**
            #    누가 나중에 `do_sample=True` 로 바꾸면 그 값들이 **조용히 적용되고**,
            #    같은 질의가 실행마다 달라진다 — 원인을 찾기 매우 어렵다.
            #    설정을 비워 두는 것과 명시적으로 끄는 것은 다르다 (D-69 · D-58).
            temperature=None,
            top_p=None,
            top_k=None,
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
            # **429 는 실패가 아니라 기다리라는 신호다.** 기본 2회로는 60건 × 6회를
            # 못 버틴다 — 2026-08-02 Gemini 실측에서 절반 이상이 RateLimitError 로
            # 폴백했고, 그 실행은 *"LLM 을 붙였다"* 고 말할 수 없는 결과가 됐다.
            max_retries=_MAX_RETRIES,
            timeout=_TIMEOUT_S,
        )

    def _thinking_off(self) -> dict:
        """**사고 모드를 끈다** — 안 끄면 짧은 태스크가 빈 문자열을 돌려준다.

        2026-08-02 실측 (Gemini 3.5 Flash · `max_tokens=16`) —

            intent 허용목록 밖: '' → 'unknown'      ← 라벨이 아니라 **빈 문자열**

        사고 토큰이 `max_tokens` 를 먼저 다 쓰고 **본문이 시작되기 전에 잘린다.**
        ①분류(16)·④검증처럼 출력이 짧은 태스크가 통째로 죽는다. Qwen3 로컬에서
        만난 것과 **같은 함정**이고(그쪽은 `enable_thinking=False`), 여기서는
        `reasoning_effort` 다.

        ⚠️ **OpenAI 본가에는 이 인자를 보내지 않는다.** 추론 모델이 아닌 모델에
        보내면 400 이 난다. 엔드포인트를 보고 판단한다 — 완벽하진 않지만,
        모르는 인자를 무조건 보내는 것보다 낫다.
        """
        if self._base_url and "generativelanguage.googleapis.com" in self._base_url:
            return {"reasoning_effort": "none"}
        return {}

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str:
        from ..prompts import build_messages

        _throttle()
        resp = self._client().chat.completions.create(
            model=self._model,
            messages=build_messages(task, user_input),  # type: ignore[arg-type]
            max_tokens=max_tokens,
            temperature=0,
            **self._thinking_off(),
        )
        return resp.choices[0].message.content or ""

    def run_raw(self, system: str, user_input: str, *, max_tokens: int = 512) -> str:
        """**5태스크 밖**의 호출 (답변 생성 · 트리아지 판정).

        05 §4 가 LLM 에 맡긴 5태스크는 파인튜닝 대상이고 04 §3 이 지표를 잰다.
        그 밖의 호출이 태스크를 빌려 쓰면 **무엇을 잰 건지 모르게 된다.**
        그래서 시스템 프롬프트를 직접 받는 문을 따로 둔다.

        ⚠️ D-36 — 여기로 나가는 입력도 개인정보 필터를 통과한 것이어야 한다.
        """
        _throttle()
        resp = self._client().chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_input},
            ],
            max_tokens=max_tokens,
            temperature=0,
            **self._thinking_off(),
        )
        return resp.choices[0].message.content or ""


class LangChainClient:
    """**LangChain 으로 LLM 을 연동한다** (필수 산출물 · D-71).

        · LangChain을 활용하여 벡터데이터베이스와 LLM 연동

    ⚠️ 2026-08-02 점검에서 `langchain` 이 소스 어디에서도 **쓰이지 않는다**는 것이
    드러났다 — 쓰이는 것은 `langgraph`(오케스트레이션)뿐이고, LLM 은 `openai` SDK 를,
    벡터DB는 `chromadb` 를 직접 부르고 있었다. `config.py` 의 `langchain_api_key`
    한 줄이 전부였다. D-64 와 같은 모양이다 — **이름만 있고 물건이 없었다.**

    ## 무엇을 LangChain 으로 하고 무엇을 안 하나 (D-71)

        LLM 연동    ✅ 이 클래스. `langchain_core` 의 메시지 타입 + `ChatOpenAI`
        벡터DB      ❌ `retrieval/store.py` 가 chromadb 를 직접 쓴다

    벡터DB를 추상 뒤에 두지 않은 것은 **게을러서가 아니라 그래야 할 것이 있어서**다 —
    `filter_by_threshold`(D-46 실측 임계값) · `dedupe_by_substance` ·
    `to_chroma_where`(빈 필터 → `EmptyFilter`, D-56) · `publisher`·`locator` 메타
    (D-37 인용 요건) · `Hit.merged_sources`. **D-46 은 실측에서 나온 우리 도메인의
    사실이고, 그런 판단은 추상 뒤에서 못 한다.**

    ## `APIClient` 와 무엇이 다른가

    **같은 모델을 부른다.** 다른 것은 *무엇을 거쳐 부르는가* 뿐이다.
    그래서 둘의 결과가 같아야 하고, 같다는 것이 곧 `LLMClient` 프로토콜이
    실제로 갈아 끼울 수 있는 추상이라는 증거다 (D-21 · 04 §5).
    """

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None):
        self.name = f"langchain:{model}" + (f"@{base_url}" if base_url else "")
        self._model = model
        self._base_url = base_url

    def _chat(self, max_tokens: int):
        from langchain_openai import ChatOpenAI

        from ...config import get_secrets

        key = get_secrets().openai_api_key
        # 사고 끄기·재시도는 `APIClient` 와 **같은 규칙**을 쓴다 (D-22).
        # 연동 방식만 다르고 조건은 같아야 A 와 A-LC 를 비교할 수 있다.
        extra = APIClient(self._model, self._base_url)._thinking_off()  # noqa: SLF001
        return ChatOpenAI(
            model=self._model,
            base_url=self._base_url,  # None 이면 OpenAI 본가
            api_key=key.get_secret_value() if key else "sk-none",  # type: ignore[arg-type]
            temperature=0,  # 04 §8 — 평가 재현성
            max_tokens=max_tokens,  # type: ignore[call-arg]
            max_retries=_MAX_RETRIES,
            timeout=_TIMEOUT_S,
            # `model_kwargs=` 로 넘기면 langchain 이 *"명시적으로 넘기라"* 고 경고한다 —
            # `reasoning_effort` 를 인자로 직접 받기 때문이다. 경고를 무시하지 않는다.
            **extra,
        )

    @staticmethod
    def _to_messages(pairs: list[dict[str, str]]) -> list:
        """`prompts.build_messages` 의 dict → `langchain_core` 메시지.

        **프롬프트의 단일 출처는 `models/prompts.py` 다** (D-22). 여기서 다시 쓰지
        않는다 — LangChain 을 쓴다고 프롬프트가 갈라지면 04 §3 이 태스크별로
        재는 지표가 **무엇을 잰 건지 모르게 된다.**
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        kinds = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
        return [kinds.get(m["role"], HumanMessage)(content=m["content"]) for m in pairs]

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str:
        from ..prompts import build_messages

        _throttle()
        out = self._chat(max_tokens).invoke(self._to_messages(build_messages(task, user_input)))
        return str(out.content)

    def run_raw(self, system: str, user_input: str, *, max_tokens: int = 512) -> str:
        """5태스크 **밖**의 호출. 파인튜닝 태스크를 빌려 쓰지 않는다 (04 §3)."""
        pairs = [{"role": "system", "content": system}, {"role": "user", "content": user_input}]
        _throttle()
        out = self._chat(max_tokens).invoke(self._to_messages(pairs))
        return str(out.content)


class EchoClient:
    """테스트용. 모델 없이 그래프·계약을 돌린다."""

    name = "echo"

    def __init__(self, responses: dict[Task, str] | None = None):
        self._responses = responses or {}

    def run(self, task: Task, user_input: str, *, max_tokens: int = 512) -> str:
        return self._responses.get(task, "")

    def run_raw(self, system: str, user_input: str, *, max_tokens: int = 512) -> str:
        return self._responses.get(system, "")  # type: ignore[call-overload]
