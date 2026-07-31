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

## 자료를 받으려면

팀 내부 배포본(`data_work_*.zip`)을 받아 이 디렉터리에 풀고, 검증한다.

```bash
python scripts/verify_corpus.py
```

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
