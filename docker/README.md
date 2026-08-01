# docker/

```
train.Dockerfile   GPU 학습 이미지 (Qwen3-4B QLoRA)
```

API 이미지는 루트의 `Dockerfile` 이다. **학습과 API를 한 이미지에 담지 않는다** —
torch·CUDA가 들어가면 이미지가 수 GB가 되고, API만 띄우는 팀원에게는 필요 없다.

```bash
docker compose up db                    # MySQL 만 — 팀원 PC 에 설치하지 않는다 (D-48)
docker compose up                       # API + MySQL
docker compose --profile train run --rm trainer --data data/train/samples.jsonl
```

`initdb/` 는 없앴다. **pgvector 확장을 만드는 스크립트였는데 D-44 로 벡터DB 가
Chroma(파일 기반)로 확정되어 쓸 일이 없어졌다.** MySQL 에서는 실행되지도 않는다.
테이블은 SQLAlchemy 가 만든다 — `python -m pettriage.app.init_db`.

`.dockerignore` 가 `data/` 를 막는다 — **이미지에 자료가 구워지면 배포가 곧 유출**이다 (D-29).
