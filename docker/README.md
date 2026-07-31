# docker/

```
train.Dockerfile   GPU 학습 이미지 (Qwen3-4B QLoRA)
initdb/            pgvector 확장 생성 — 컨테이너 최초 기동 시 1회
```

API 이미지는 루트의 `Dockerfile` 이다. **학습과 API를 한 이미지에 담지 않는다** —
torch·CUDA가 들어가면 이미지가 수 GB가 되고, API만 띄우는 팀원에게는 필요 없다.

```bash
docker compose up                       # API + pgvector
docker compose --profile train run --rm trainer --data data/train/samples.jsonl
```

`.dockerignore` 가 `data/` 를 막는다 — **이미지에 자료가 구워지면 배포가 곧 유출**이다 (D-29).
