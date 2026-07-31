# data/train/

멀티태스크 학습 샘플 (JSONL). **커밋하지 않는다.**

```
samples.jsonl   TrainSample 직렬화 1건/행 — models/datasets/schema.py 참조
```

생성은 03 §3(distillation + 사람 검수)을 따르며, 학습 진입 시
**골든셋 누수 검사**가 먼저 돌아 학습셋과 평가셋이 겹치면 중단된다.

```bash
PETTRIAGE_PROFILE=train make train
```
