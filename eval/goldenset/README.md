# eval/goldenset/ — 골든셋 100건

설계: [`docs/04_테스트-평가계획.md`](../../docs/04_테스트-평가계획.md) §2
작성법: **[`docs/04a_골든셋작성지침.md`](../../docs/04a_골든셋작성지침.md)** — 쓰기 전에 먼저 읽는다

```
골든셋_양식.csv   복사해서 쓰는 원본 + 예시 9건. 여기에 직접 쓰지 않는다
golden_lse.csv    이서은   G-001 ~ G-025   ← 기준선. 먼저 쓴다
golden_ohb.csv    오한빈   G-101 ~ G-125
golden_lgj.csv    이근준   G-201 ~ G-225
golden_ksr.csv    권소라   G-301 ~ G-325
```

**넷이 동시에 시작하지 않는다.** 이서은이 10건을 쓰고 팀장이 검수해 기준을 확정한 뒤
나머지가 착수한다. 기준이 나중에 바뀌면 다 다시 써야 한다.

**데이터 적재를 기다리지 않고 지금 쓸 수 있다.** 질문과 기대 동작은 설계 문서만으로 정해진다.

## 커밋 전에

```bash
python scripts/check_goldenset.py eval/goldenset/golden_ohb.csv
```

ERROR 0 이어야 커밋한다. 검사가 잡는 것:

- 필수 칸 결측 · 허용값 오타
- **`answered` 인데 `must_cite` 가 빈 행** — 근거 없는 정답은 채점이 안 된다
- **`refused` 인데 사유가 없는 행** — 오류 분석에 쓸 수 없다
- **`clarify`·`refused` 인데 등급이 적힌 행**
- **`MONITOR` 정답인데 상승 조건 문구가 없는 행** (D-39)
- **`must_cite` 에 대장(`data/manifests/`)에 없는 `source_id` 를 적은 행** — 그 케이스는 영원히 통과할 수 없다
- 종별 최소 건수 · `clarify`/`refused` 비율 (04 §2.2 · §2.3)
- `case_id`·질문 중복 · 검수자 미기입

## 배분 (04 §2.2 · §2.3)

| 유형 | 비중 | | 종 | 최소 |
|---|---|---|---|---|
| 단순 사실형 | 25% | | 개 | 35 |
| 용량·개체 조건형 | 20% | | 고양이 | 30 |
| 기록 참조형 | 15% | | **앵무새** | **30** |
| 종 구분형 | 15% | | | |
| **슬롯 결측형** → `clarify` | **10%** | | | |
| **답 없음** → `refused` | **15%** | | | |

`answered` 만 모으면 이 시스템의 특징인 **거절과 되묻기**가 평가되지 않는다.
검사기가 `clarify` 10% · `refused` 15% 미만이면 경고한다.

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

- **`MONITOR` 정답은 상승 조건을 함께 요구한다.** 조건 없는 "관찰"은 과소평가로 채점 (04 §4.1.0)
- 조류에는 수치 조건이 거의 없다. **질적 조건**으로 정답을 구성한다 (D-09 개정)
- 고양이 단독 자료는 2단계뿐이라 `mammal`·`all` 근거를 함께 인용해야 한다 (D-39)

## 검수는 교차한다 (04 §2.4)

| 작성 | 검수 |
|---|---|
| 이서은 | **오한빈** ← 기준 확정을 겸한다 |
| 오한빈 | 이서은 |
| 이근준 | 권소라 |
| 권소라 | 이근준 |

자기가 쓴 케이스를 자기가 검수하면 같은 오해가 두 번 통과한다.
