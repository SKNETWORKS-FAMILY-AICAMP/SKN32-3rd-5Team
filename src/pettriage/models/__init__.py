"""멀티태스크 sLLM — Qwen3-4B QLoRA (WS3).

설계 근거: docs/03_모델-멀티태스크학습.md · docs/06 D-05 · D-42

```
tasks.py       태스크 5종 정의 — 그래프 노드·지표와 1:1 대응
prompts.py     프롬프트 템플릿 — 학습과 추론이 같은 문자열을 쓴다
datasets/      학습 샘플 스키마 · 태스크 혼합 · 누수 검사
training/      QLoRA 학습 (HF PEFT + TRL)
serving/       LLMClient 프로토콜 — 로컬 Qwen · API · Echo
```

torch·transformers 는 **함수 안에서** 임포트한다.
GPU 없는 팀원이 API만 띄울 수 있어야 한다 (`pip install -e '.[api]'`).
"""

from .tasks import DEFAULT_TASKS, SPECS, Task

__all__ = ["DEFAULT_TASKS", "SPECS", "Task"]
