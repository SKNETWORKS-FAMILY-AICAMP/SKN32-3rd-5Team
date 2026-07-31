# PetTriage — 반려동물 헬스케어 다이어리 & 응급 대응

> **RAG가 근거를 가져오고, 파인튜닝한 sLLM이 그 근거가 지켜졌는지 검증한다.**
> 대상 종 — 개 · 고양이 · **앵무새**

SKN 3차 단위 프로젝트 · 팀 **save the pet**
팀장 **오한빈** · 팀원 이근준 · 권소라 · 이서은

---

## 이 시스템이 하는 일

보호자가 *"우리 강아지가 초콜릿 먹었는데 괜찮아?"* 라고 물으면,
**진단하지 않고** — 공적 자료의 근거를 찾아 **지금 무엇을 해야 하는지**에 답한다.

```
① 분류 → ② 슬롯 추출 → 검색 + 계산 → ③ 압축 → 생성 + 트리아지 → ④ 근거 검증 → ⑤ 평이화
              │
              └─ 체중·섭취량·종이 없으면 추측하지 않고 되묻는다 (상한 2회)
```

일상 기록(다이어리)이 **내부 문서**가 되어, *"내 기록 × 공적 기준 → 판정"* 구조가 성립한다.

**베이스 sLLM은 Qwen3-4B**이며 QLoRA(4bit)로 태스크 5종을 함께 학습한다 (D-42).
대형 LLM 폴백 경로는 같은 `LLMClient` 프로토콜 뒤에 유지된다 — 파인튜닝이 실패해도 시연은 남는다 (D-21).

## 세 가지 안전 장치 — 지표가 아니라 구조로 막는다

| | |
|---|---|
| 🔒 **트리아지 하향 금지 게이트** | `triage_level = max(rule, llm)` — **LLM은 등급을 낮출 수 없다** |
| **종 미확인 시 답변 금지** | 포유류 기준의 조류 적용은 치명적이다. 종이 없으면 검색 자체를 하지 않는다 |
| **근거 없으면 답하지 않는다** | 문장별 근거 검증 → 없는 문장 제거 → 재검색 1회 → 그래도 부족하면 **거절** |

트리아지 등급은 임의로 정하지 않고 **코퍼스의 행동 지시어에서 도출**했다.

| 값 | 정수 | 배지 | 사용자에게 |
|---|---|---|---|
| `EMERGENCY` | 4 | 응급 | 지금 바로 동물병원으로 가세요 |
| `CALL_NOW` | 3 | 전화 | 지금 수의사에게 전화해 상태를 알리세요 |
| `VISIT_SOON` | 2 | 내원 | 오늘 중 진료를 받으세요 |
| `MONITOR` | 1 | 관찰 | 집에서 지켜보고, 아래 증상이 나타나면 연락하세요 |

숫자가 클수록 위험하므로 `max()` 가 그대로 성립한다. → [`triage/levels.py`](src/pettriage/triage/levels.py)

---

## 시작하기

GPU가 없어도 API·테스트는 전부 돈다. 학습 의존성은 `[train]` 으로 분리되어 있다.

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
make install                                          # = pip install -e '.[api,rag,ingest,dev]' -c constraints.txt
cp .env.example .env                                  # 키 입력
make test                                             # 68 tests
make serve                                            # http://127.0.0.1:8000
```

컨테이너로 띄우려면 `make up` (API + pgvector). GPU 학습은 `make train`.

| 설치 그룹 | 언제 |
|---|---|
| `.[api]` | 배달 계층만. 팀원 대부분이 이것으로 충분 |
| `.[rag]` | 임베딩·벡터DB·LangGraph |
| `.[ingest]` | PDF·HTML 파싱, EXIF 제거 |
| `.[train]` | **GPU 전용** — torch·peft·trl·bitsandbytes |
| `.[dev]` | pytest·ruff·pre-commit |

`constraints.txt` 가 버전을 고정한다. **실험 결과를 보고할 때 이 파일의 커밋 해시를 함께 적는다** (04 §8).

`make serve` 는 API와 데모 프론트를 **같은 출처에서** 띄운다. 빌드 도구도 node도 없다.
벡터DB·LLM 없이도 되묻기·거절·하향 금지 게이트가 실제로 동작한다 (`engine: stub`).

## API — WS2와 WS5의 합의문

계약은 [`app/contracts.py`](src/pettriage/app/contracts.py) 한 곳이고, 스펙은 `/docs` 에서 볼 수 있다.

| 엔드포인트 | 용도 |
|---|---|
| `POST /api/ask` | 질의응답. **항상 200**, `status` 로 분기 |
| `POST /api/records` · `GET /api/report` | 다이어리 기록 · 기간 리포트 |
| `GET /api/triage-levels` | 등급 정의 **+ 도출 근거 원문** — 프론트가 등급 표현을 하드코딩하지 않게 한다 |
| `GET /api/health` | 현재 엔진 (`stub` / `graph`) |

```
status = answered   근거를 찾아 판정했다   → 배지 + 근거 + 감사정보(규칙/LLM/차단여부)
         clarify    슬롯이 비어 되묻는다   → 최대 2회, 초과하면 refused 로 전환
         refused    근거없음·판정불가      → 사유 + 수의사 상담 권고
```

**거절을 4xx로 내보내지 않는다.** 거절은 장애가 아니라 설계된 경로이고(02 §9),
4xx로 만들면 프론트가 에러 핸들러에서 처리해 거절 화면이 장애 화면처럼 보인다.

정책을 문서가 아니라 **스키마로** 강제한 지점 세 곳:

- `answered` 인데 `citations` 가 비면 **응답 객체 생성이 실패한다** — 근거 없는 답은 만들 수 없다
- `MONITOR` 인데 상승 조건이 없으면 실패한다 — 조건 없는 "관찰"은 과소평가다 (D-39)
- 경로 ②(사실추출) 근거에 원문 인용을 실으면 실패한다 (D-37)

엔진 교체는 [`app/deps.py`](src/pettriage/app/deps.py) **한 줄**이다.
WS2의 LangGraph가 완성되면 `StubEngine` → `GraphEngine` 으로 바꾼다.
계약·프론트·테스트는 손대지 않는다.

## 저장소 구조

```
configs/         ⚙️ 재현에 필요한 값 전부 (모델·학습·검색·서빙). 커밋된다
docs/            설계 문서 13종 — 여기부터 읽는다
src/pettriage/
  paths.py       프로젝트 루트 탐색 — 설치 형태와 무관하게 configs/·web/ 을 찾는다
  config.py      YAML + 환경변수 로딩 (못 찾으면 크게 실패한다)
  schemas.py     Fact · Chunk
  ingest/        수집 → 사실 추출 → 템플릿 문장화 → 청킹   (WS1)
  retrieval/     임베딩 · 벡터DB · 검색                    (WS2)
  compute/       비-RAG 계산 노드 (에너지·독성 임계치)
  triage/        등급 정의 + 하향 금지 게이트  🔒
  graph/         LangGraph 상태·노드                      (WS2)
  models/        멀티태스크 sLLM                          (WS3)
    tasks.py       태스크 5종 — 그래프 노드·지표와 1:1
    prompts.py     학습·추론 공용 템플릿
    datasets/      샘플 스키마 · 태스크 혼합 · 누수 검사
    training/      Qwen3-4B QLoRA (PEFT + TRL)
    serving/       LLMClient — 로컬 Qwen · API · Echo
  privacy/       개인정보 제거 (EXIF·얼굴·필드)            (D-36)
  app/           FastAPI — 계약 · 라우터 · 세션 · 저장소    (WS5)
  tools/         운영 도구 (코퍼스 검증) — 콘솔 스크립트 대상
web/             데모 프론트 (정적 HTML 1장)               (WS5)
eval/            골든셋 · 평가 하네스 · 결과 보고서         (WS4)
scripts/         셸 진입점 (얇은 래퍼)
docker/          학습 이미지 · DB 초기화
tests/           안전 장치 회귀 테스트
notebooks/       탐색용 (결론은 여기 남기지 않는다)
data/            🔴 매니페스트만 커밋. 자료 파일은 커밋 금지
artifacts/       🔴 학습 어댑터. 커밋 금지
```

폴더마다 `README.md` 가 있어 **거기에 무엇이 들어가고 무엇이 들어가면 안 되는지**를
그 자리에서 알 수 있다. 협업 규약은 [`CONTRIBUTING.md`](CONTRIBUTING.md).

**설정과 비밀을 나눈 기준** — `configs/*.yaml` 은 재현에 필요하므로 커밋하고,
`.env` 는 환경마다 다르거나 비밀이므로 커밋하지 않는다.
파라미터가 코드에 흩어져 있으면 *"그때 top-k가 몇이었더라"* 에 답할 수 없다 (04 §8).

```bash
PETTRIAGE_PROFILE=eval make test        # configs/eval.yaml 이 default 를 덮는다
PETTRIAGE__RETRIEVAL__TOP_K=8 make serve  # 파일을 고치지 않고 한 번만
```

## 문서 — 읽는 순서

| 순서 | 문서 | 내용 |
|---|---|---|
| 1 | [`05_설계원칙`](docs/05_설계원칙-코드와LLM의분업.md) | **두 축** · 3문 · 정지 판정 · 안티패턴 — *왜 이렇게 만드는가* |
| 2 | [`00_기획`](docs/00_기획-요구사항분석.md) | 과제 해석 · 도메인 · 로드맵 · 팀 — *전체 지도* |
| 3 | [`02_아키텍처`](docs/02_시스템-아키텍처.md) | 파이프라인 4종 · LangGraph · 검색 정책 |
| 4 | [`06_결정기록`](docs/06_설계결정기록.md) | **결정 33건**의 맥락·대안·트레이드오프 |

데이터: [`01`](docs/01_데이터-수집및전처리.md) 방침 · [`01a`](docs/01a_자료분석보고.md) 분석 · [`01b`](docs/01b_자료검증보고.md) 검증 · [`01c`](docs/01c_데이터-작업지시.md) 작업지시 · [`01d`](docs/01d_자료보관규칙.md) 보관
모델: [`03`](docs/03_모델-멀티태스크학습.md) · 평가: [`04`](docs/04_테스트-평가계획.md)

> **"단점은 없나요?"** → 06의 트레이드오프 문장
> **"자료를 왜 버렸나요?"** → [`data/manifests/DELETION_LOG.csv`](data/manifests/DELETION_LOG.csv)

---

## 🔴 데이터는 이 저장소에 없다

**이 저장소는 공개이고, 코퍼스에는 이용약관 제약이 있는 자료가 섞여 있다.**
따라서 자료 파일은 **커밋하지 않는다.** `data/manifests/` 의 대장 4종만 올린다.

| 대장 | 내용 |
|---|---|
| `MANIFEST.csv` | 팀이 전달한 원본 |
| `SNAPSHOT_MANIFEST.csv` | 웹·PDF 텍스트 스냅샷 (품질 5등급 판정 포함) |
| `SOURCES_CITED.csv` | **원문을 담지 않는 출처**의 인용 정보 |
| `DELETION_LOG.csv` | 삭제 이력 — SHA-256 해시 · 약관 원문 · 판정 근거 |

### 어떻게 이렇게 됐나

수집한 62건에 **수집 전 게이트**를 소급 적용했다 (30개 도메인 약관 전수 확인).

| 판정 | 건수 | 뜻 |
|---|---|---|
| 통과 | **8** | 원문 청크 적재 가능 (FDA · Frontiers CC BY · 농진청) |
| 사실추출 한정 | **38** | 원문 미적재. **사실만 뽑아 우리 문장으로** 재구성 |
| ⛔ 삭제 | **15** | 약관이 AI 활용 자체를 금지 |

> *"use of **artificial intelligence** … to rewrite, adapt, or **repurpose** this content"* — 삭제 근거 중 하나

**원문을 그대로 적재할 수 있는 자료가 8건뿐이라, 경로 ②(사실 추출 + 문장화)가 기본이다.**
그 결과 문장화 자체가 검증 대상이 되었고, 평가에 **층 0(데이터)** 을 신설했다.

```
원문 → [사실 표 추출] → [템플릿 문장 생성 — 코드] → 청킹 → 벡터DB
              ↑ 여기만 검증하면 된다 (문장이 아니라 필드)
```

**문장화는 LLM이 아니라 코드가 한다.** → [`ingest/verbalize.py`](src/pettriage/ingest/verbalize.py) · [`templates.py`](src/pettriage/ingest/templates.py)

---

## 개발 규칙

- 새 판단 지점이 생기면 **3문**(docs/05 §1)을 먼저 적용하고 결과를 문서에 남긴다
- LLM에 맡기기로 했다면 **같은 PR에 검증 코드를 함께** 넣는다
- 안전 관련 조정은 **한 방향(위험도 상향)으로만** 허용한다
- **자료를 수집하기 전에** docs/05 §8.1 게이트를 통과시킨다 — 받고 나서 버리지 않는다
- `data/` 아래 자료 파일을 커밋하지 않는다. `git add -f` 를 쓰지 않는다
- **파라미터를 코드에 박지 않는다.** `configs/*.yaml` 로 뺀다 — 재현성 요건이다
- 무거운 임포트(torch·transformers)는 **함수 안에서** 한다. GPU 없는 팀원과 CI가 깨진다

## 라이선스

코드는 [MIT](LICENSE). **데이터는 이 라이선스의 적용을 받지 않는다** — 각 자료의
출처 이용 조건을 따르며, 조건은 `data/manifests/SOURCES_CITED.csv` 에 기록되어 있다.

본 프로젝트는 **비상업 교육·연구 목적**이며, 의료 행위를 대체하지 않는다.
