# API 이미지 — 배달 계층만 담는다 (D-40).
#
#   학습(torch·CUDA)은 이 이미지에 넣지 않는다. 이미지가 수 GB 커지고,
#   API를 띄우는 팀원 대부분에게 필요 없다. 학습은 docker/train.Dockerfile.
#
#   빌드:  docker build -t pettriage-api .
#   실행:  docker compose up

FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1

COPY pyproject.toml constraints.txt README.md LICENSE ./
COPY src ./src

# 의존성 레이어를 소스와 분리해 캐시가 살아있게 한다
RUN pip install --prefix=/install '.[api]' -c constraints.txt


FROM python:3.11-slim AS runtime

# 루트로 돌리지 않는다
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder /install /usr/local
COPY src ./src
COPY web ./web
COPY configs ./configs
COPY scripts ./scripts

# PETTRIAGE_ROOT 를 명시한다 — 파일 위치로 루트를 추측하게 두지 않는다.
# configs/ 를 못 찾으면 config.py 가 ConfigNotFound 로 크게 실패한다 (조용한 폴백 금지).
ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PETTRIAGE_ROOT=/app \
    PETTRIAGE_PROFILE=default

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

# 컨테이너에서는 0.0.0.0 에 바인딩한다 (설정 기본값은 127.0.0.1)
CMD ["uvicorn", "pettriage.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
