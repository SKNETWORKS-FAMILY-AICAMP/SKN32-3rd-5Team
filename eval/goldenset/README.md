# 골든 셋

설계: [`docs/04_테스트-평가계획.md`](../../docs/04_테스트-평가계획.md) §2

**100건 이상.** 시나리오 기반 — (가상 프로필 + 누적 기록) + 질의 →
기대 트리아지 등급 · 근거 문서(`chunk_id`) · 정답 요지.

## 스키마

```csv
case_id,species,scenario,query,expected_triage,expected_chunk_ids,expected_gist,
evidence_quote,author,reviewer,notes
```

- `expected_triage` — `EMERGENCY` / `CALL_NOW` / `VISIT_SOON` / `MONITOR` (D-39)
- `evidence_quote` — **정답 등급을 정한 근거 문장.** 작성자가 임의로 매기지 않는다
- `author` ≠ `reviewer` — 분리 기록 (§2.4)

## 배분 (§2.2 · §2.3)

| 유형 | 비중 | | 종 | 최소 |
|---|---|---|---|---|
| 단순 사실형 | 25% | | 개 | 35 |
| 용량·개체 조건형 | 20% | | 고양이 | 30 |
| 기록 참조형 | 15% | | **앵무새** | **30** |
| 종 구분형 | 15% | | | |
| 슬롯 결측형 | 10% | | | |
| **답 없음** | 15% | | | |

## 필수 포함 — 출처 상충 사례

코퍼스 실측으로 이미 확보된 것들이다. **하향 금지 게이트의 동작을 직접 검증한다.**

| 사례 | 출처 A | 출처 B | 정답 |
|---|---|---|---|
| **발작** | AAHA S-037 *"as soon as the seizure ends, immediately contact"* → `CALL_NOW` | FOUR PAWS S-030 *"if… convulsing… go immediately"* → `EMERGENCY` | **`EMERGENCY`** |
| 포도(개) | S-021 *"even small amounts emergencies"* | S-063 *"a grape or two with no problem"* | 상위 채택 |
| 사과씨(조류) | S-005 `NEVER` | S-071 *"an occasional apple seed will not harm"* | 상위 채택 |
| 아보카도 | FDA S-029 *"only mildly toxic to dogs and cats"* | Banfield S-085 무조건 독성 | 상위 채택 |
| 영양 기준 | FEDIAF 라이신 0.85 | AAFCO 2.08 (2.4배) | 복수 근거 제시 |

## 주의

- **`MONITOR` 정답은 상승 조건을 함께 요구한다.** 조건 없는 "관찰"은 과소평가로 채점 (§4.1.0)
- 조류에는 수치 조건이 거의 없다. **질적 조건**으로 정답을 구성한다
- 고양이 단독 자료는 2단계뿐이라 `mammal/`·`all/` 근거를 함께 인용해야 한다
