# data/ — 이 디렉터리에는 자료 파일이 없다

**이 저장소는 공개이고, 코퍼스에는 이용약관 제약이 있는 자료가 섞여 있다.**
따라서 자료 파일은 커밋하지 않는다. git에 올라가는 것은 `manifests/` 의 대장 4종뿐이다.

근거: [`docs/06_설계결정기록.md`](../docs/06_설계결정기록.md) D-29 · D-33 · D-37
보관 규칙 전문: [`docs/01d_자료보관규칙.md`](../docs/01d_자료보관규칙.md)

## 대장 4종

| 파일 | 내용 |
|---|---|
| `manifests/MANIFEST.csv` | 팀이 전달한 원본 (해시·라이선스·수집자) |
| `manifests/SNAPSHOT_MANIFEST.csv` | 웹·PDF 텍스트 스냅샷 42건 + **품질 5등급 판정** |
| `manifests/SOURCES_CITED.csv` | 원문을 담지 않는 출처의 인용 정보 |
| `manifests/DELETION_LOG.csv` | 삭제 15건 — **SHA-256 · 약관 원문 · 판정 근거** |

## 로컬 디렉터리 (전부 gitignore)

```
raw/          원본 바이너리 — 불변
snapshot/     웹·PDF 텍스트 추출본 — 불변. <종>/ 로 분류
extracted/    텍스트 추출 결과
structured/   사실 표 (CSV / JSONL)
indexed/      문장화 청크 · 적재 입력
```

`raw·snapshot → extracted → structured → indexed`

## 자료를 받으려면 — **먼저 필요한지부터 보라**

**대부분은 필요 없다.** 이 zip 없이도 개발이 된다.

| 하려는 일 | zip 이 필요한가 | 대신 필요한 것 |
|---|---|---|
| **UI·프론트** | ❌ | `pip install -e '.[api]'` → `make serve` |
| API 연동 | ❌ | `http://127.0.0.1:8000/docs` · `app/contracts.py` |
| **벡터DB 적재·검색** | ❌ | `data/facts/facts_ohb.csv` — **이미 저장소에 있다** |
| 백엔드(계정·프로필) | ❌ | `make db` (MySQL 컨테이너) |
| 층 0 검증 `make verify` | ✅ | — |
| 골든셋 채점 기준 대조 | ✅ | 원문이 채점표다 (D-29) |
| 사실 표 행 추가·수정 | ✅ | 원문 대조 없이 고치지 않는다 |

> **인덱스의 입력은 원문이 아니라 사실 표다** (D-37 경로②).
> `python scripts/build_index.py --store chroma` 는 `data/facts/` 만 읽는다 —
> **검색 개발은 이 zip 없이 진행된다.**

정말 필요하면 **팀장에게 요청한다.** `data_work_*.zip` 은 공개 저장소에 올리지 않으므로
링크가 문서에 없다 (D-29). 받은 뒤:

```bash
# 저장소 루트에서 (날짜는 받은 파일 이름에 맞춘다)
unzip data_work_*.zip
make verify          # = python scripts/verify_corpus.py
```

`make verify` 의 **[대장 ↔ 실파일] 절이 비어 있어야** 제대로 풀린 것이다 —
*"대장에 있으나 파일 없음"* 도 *"파일은 있으나 대장에 없음"* 도 0건이라는 뜻이다.

### ⚠ 실행 방법 세 가지가 **같지 않다**

| 명령 | 어느 코드를 쓰나 |
|---|---|
| **`make verify`** (= `python scripts/verify_corpus.py`) | **저장소의 `src/`** — 래퍼가 `sys.path` 앞에 넣는다 |
| `pettriage-verify` | **설치된 패키지** |
| `python -m pettriage.tools.verify_corpus` | **설치된 패키지** |

아래 둘은 `pip install -e` 가 다른 경로를 가리키고 있으면 **엉뚱한 루트를 잡는다.**
그러면 자료를 제대로 풀었는데도 *"자료 파일이 로컬에 없다"* 가 뜬다.

> **그래서 `make verify` 를 쓴다.** 래퍼가 저장소 코드를 강제하므로 이 함정이 없다.
> 다른 방법으로 돌리다 위 증상을 만나면 `python -c "import pettriage; print(pettriage.__file__)"` 로
> 확인하고, 저장소 밖 경로가 찍히면 `pip install -e '.[api,dev]' -c constraints.txt` 를 다시 한다.
>
> 어느 쪽이든 **거짓 통과는 나지 않는다** — `paths.py` 가 못 찾으면 조용히 추측하지 않고
> **검사가 축소되었다고 보고**한다 (04 §8).

**작업용 배포본은 팀 내부 한정이며 외부 배포하지 않는다** (D-29).
제출용은 배포 가능분만 담긴 별도 아카이브를 쓴다.

## 자료를 추가하려면

**먼저** [`docs/05_설계원칙-코드와LLM의분업.md`](../docs/05_설계원칙-코드와LLM의분업.md) §8.1의
**수집 전 게이트**를 통과시킨다. 받고 나서 버리지 않는다.

- ☐ 약관에 **용도 금지** 조항이 있는가? `artificial intelligence` · `train` · `data mining` · `repurpose`
- ☐ **AI를 언급하지 않고 막는 표현**은? `information retrieval system` · `stored, processed` · `store` · `derivative works`
- ☐ 개인·비상업 **복제 허락 문언**이 있는가
- ☐ 구독·로그인이 필요한가
- ☐ 대량 DB인가 / 개인정보가 있는가
