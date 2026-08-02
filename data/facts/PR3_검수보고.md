# PR #3 병합 전 검수 보고

> **SKN 3차 단위 프로젝트** · `ohb → main` 병합 대상 검수
> 대상: 이근준 `lgj`(백엔드 424줄) · 이서은 `lse`(골든셋 25건)
> 검수일 2026-08-01 (KST) · **검수자: 오한빈 (팀장)**

---

## 0. 결론

| | 판정 |
|---|---|
| **이서은 `lse`** | ✅ **병합 가능.** `must_cite` 공란은 팀장이 채운다 |
| **이근준 `lgj`** | ⛔ **차단 2건을 고친 뒤 병합** |

`main` 은 되돌리기 어렵다. 차단 항목은 **머지 후 고치면 되는 것이 아니라,
머지되면 문제가 보이지 않게 되는 것**들이다.

---

## 1. ⛔ 차단 — 의존성이 선언되지 않았다

`auth.py`·`database.py`·`routes/*.py` 가 쓰는 라이브러리 **5종 중 3종이 어디에도 없다.**

```
pyproject.toml  dependencies      pydantic · pydantic-settings · PyYAML
                [api]             fastapi · uvicorn
                [rag]             langgraph · chromadb · ...
```

```
  OK   sqlalchemy      (다른 extra 가 딸려 들여옴 — 우연이다)
  ✗    passlib         ← 미선언
  ✗    jwt (PyJWT)     ← 미선언
  OK   bcrypt          (passlib 의 백엔드로 우연히 존재)
  ✗    pymysql         ← 미선언
```

이근준 팀원 PC에서는 손으로 `pip install` 해서 돌아간다. **저장소를 받은 사람에게는 안 돈다.**
04 §8 재현성 조항에 정면으로 걸린다.

### 그리고 이건 조용히 실패한다 ← 진짜 문제

`routes/__init__.py` 가 `ImportError` 를 잡아서 로그만 남기고 넘어간다.

```
$ DATABASE_URL="mysql+pymysql://..." python -c "from pettriage.app.routes import ALL_ROUTERS"
WARNING:pettriage.app.routes:DB 라우터 로드 실패 — auth/pets 비활성화: No module named 'jwt'
라우터 3 개
```

**앱이 정상 기동한다.** 회원가입도 로그인도 없는 채로.
`DATABASE_URL` 을 설정한 사람은 인증이 켜졌다고 믿는데 실제로는 꺼져 있고,
알 수 있는 단서는 아무도 안 읽는 `WARNING` 한 줄뿐이다.

> `DATABASE_URL` 이 없을 때 라우터를 건너뛰는 것은 **좋은 설계다** — DB 없는 팀원과 CI에서도 앱이 뜬다.
> 문제는 그 방어가 **의존성 누락까지 함께 삼킨다**는 것이다.
> 04 §8 — *"검사 축소는 드러나야 한다."*

### 조치

```toml
# pyproject.toml
[project.optional-dependencies]
db = [
  "SQLAlchemy>=2.0,<3",
  "PyMySQL>=1.1,<2",
  "passlib[bcrypt]>=1.7.4,<2",
  "PyJWT>=2.10,<3",
  "email-validator>=2.2,<3",
]
```

그리고 `routes/__init__.py` 의 분기를 **두 경우로 쪼갠다.**

| 상황 | 지금 | 고친 뒤 |
|---|---|---|
| `DATABASE_URL` 없음 | 건너뜀 (조용히) | 건너뜀 — **의도된 것이므로 조용해도 된다** |
| `DATABASE_URL` 있는데 import 실패 | 건너뜀 (조용히) | **기동 실패.** `[db]` extra 를 설치하라고 말한다 |

CI 에도 `db-deps` 잡을 추가한다. `rag` extra 를 CI 가 설치하지 않아
transformers 4.48 이 Qwen3 를 못 읽는 걸 놓쳤던 것과 **같은 종류의 사각지대**다.

---

## 2. ⛔ 차단 — 되묻기 슬롯을 DB에 영구 저장한다

`models.py` 의 `ChatSession` 이 이 셋을 테이블 칼럼으로 들고 있다.

```python
clarify_turns: Mapped[int]
weight_kg:     Mapped[float | None]
amount_g:      Mapped[float | None]
```

그런데 같은 저장소의 `app/session.py` 는 첫 문단에 이렇게 적혀 있다.

> 메모리 구현이다. 프로세스가 죽으면 사라지는 것이 **의도**다 —
> 되묻기 슬롯(체중·섭취량)은 **보관할 이유가 없다** (D-36 최소 수집).

`graph/state.py` 도 같다.

> 이 State 는 **되묻기 세션 상태**다 — 조각 3. **휘발성이고 한 질의 안에서만 산다.**

**같은 값을 한쪽은 "휘발이 의도"라고 하고 다른 쪽은 테이블에 넣는다.**
D-36 조치 1(최소수집)·조치 7(목적 달성 시 파기)과 05 §3 조각 구분이 함께 깨진다.
체중·섭취량 시계열은 D-36 표에서 *"누적되면 생활 패턴이 드러난다"* 로 지목한 바로 그 항목이다.

### 조치

`ChatSession` 에서 **세 칼럼을 뺀다.** 대화 로그(`ChatMessage`)를 남기는 것 자체는 좋다 —
평가·오류 분석에 필요하고 D-13 과소평가율 추적의 근거가 된다.
**슬롯만 빼면 된다.**

빼지 않고 남기려면 그건 D-36 을 뒤집는 결정이므로 **D-48 로 기록해야 한다.**
어느 쪽이든 좋지만, **기록 없이 코드만 바뀌는 것은 안 된다** (D-22).

---

## 3. 🟡 고쳐서 받았다 (이미 반영, 파일 전달 완료)

| 파일 | 무엇이 | 왜 |
|---|---|---|
| `config.py`·`auth.py` | `jwt_secret_key` 기본값 `"change-me-in-production"` 제거 → `SecretStr \| None`, 없으면 `JWTKeyMissingError` | 자리표시자가 기본값이면 **아무도 안 바꾼 채 배포된다.** 키를 아는 사람은 누구나 토큰을 위조한다 |
| `routes/pets.py` | `raise ... from None` 2곳 | `except` 안에서 그냥 `raise` 하면 원본 JWT 예외가 응답에 섞인다 — 키·알고리즘이 샌다 |
| `routes/__init__.py` | EOF 개행 + `__all__` 에 개별 라우터 복원 | 기존 `from .routes import ask_router` 가 깨진다 |
| 6파일 | `ruff` 8건(E501·B904·UP035·I001) + 포맷 | **지금 상태로 push 하면 CI 빨간불** |

---

## 4. 🟡 병합 후 고칠 것

### 4.1 `datetime.utcnow()` — naive UTC

`models.py` 의 5개 칼럼이 전부 `default=datetime.utcnow` 다.

- Python 3.12 부터 **deprecated**
- **timezone 정보가 없는** datetime 이라, 표시 계층에서 KST 변환을 잊는 순간 **9시간 틀린다**

06 문서가 이미 같은 사고를 겪었다 — 전 항목 날짜가 UTC로 찍혀 하루씩 앞당겨져 있었고 8/1 에 정정했다.
**같은 종류의 실수가 DB 스키마로 들어오는 것이다.**

```python
from datetime import UTC, datetime
default=lambda: datetime.now(UTC)          # + 칼럼을 DateTime(timezone=True) 로
```

### 4.2 회원가입 레이스

```python
if db.query(User).filter(User.email == req.email).first():   # ← 확인
    raise HTTPException(409)
db.add(user); db.commit()                                     # ← 그 사이 다른 요청이 끼어들 수 있다
```

동시 가입 시 `IntegrityError` 가 **500** 으로 나간다. `commit()` 을 `try` 로 감싸고
`IntegrityError` 를 409 로 바꾼다. `unique=True` 가 있으니 데이터는 안전하다 — **응답 코드만 틀린다.**

### 4.3 `get_db()` 에 rollback 이 없다

```python
try:
    yield session
finally:
    session.close()
```

예외로 빠져나오면 트랜잭션이 열린 채 커넥션이 풀로 돌아간다.
`except: session.rollback(); raise` 를 넣는다.

### 4.4 이메일 형식 검증이 없다

`email: str = Field(min_length=3)` 이라 **`"a@"` 도 통과**한다. `EmailStr` 로 바꾼다
(위 `[db]` extra 의 `email-validator` 가 이걸 위한 것이다).

### 4.5 bcrypt 72바이트 절단

passlib+bcrypt 는 **72바이트를 넘는 부분을 조용히 버린다.**
한글은 3바이트/자라 **25자부터 뒤가 무시된다.** 긴 비밀번호를 쓴 사용자가
자기도 모르게 앞부분만 쓰게 된다. `max_length=64` 를 걸어 명시적으로 막는다.

### 4.6 죽은 필드

`User.last_login_at` 은 스키마에만 있고 **아무 데서도 갱신하지 않는다.**
`login()` 에서 채우거나 칼럼을 뺀다.

### 4.7 `(session_id, seq)` 유니크 제약 없음

`ChatMessage.seq` 가 중복될 수 있어 대화 순서가 뒤집힐 수 있다.

---

## 5. ⚪ 팀장 판단이 필요한 것

### 5.1 D-17 이 코드에서 한 번도 지켜지지 않았다

D-17 은 *"전 구간 g 단일 단위, 표시 시에만 환산"* 으로 확정돼 있다. 그런데

```
app/session.py:35      SLOTS = (..., "weight_kg", "amount_g")
graph/state.py:23      weight_kg: float
app/contracts.py:47    weight_kg: float | None
models.py (신규)        weight_kg: Mapped[float | None]
```

**전부 `weight_kg` 다.** 이근준 팀원의 잘못이 아니라 **원래부터 어긋나 있었고,
이번에 DB 스키마까지 그 방향으로 굳는 것**이다.

지금 와서 보면 **`kg` 쪽이 맞을 수 있다.** 규칙 테이블 단위가 `mg/kg`·`g/kg` 라
체중을 kg 로 들고 있는 편이 곱셈이 자연스럽다. `amount_g` 는 g 라 섞여 있는 것도 사실이다.

**어느 쪽이든 문서와 코드가 같아야 한다.** 둘 중 하나:

- (a) 코드를 D-17 대로 `weight_g` 로 통일 — 스키마·계약·State 전부 손댄다
- (b) **D-17 을 개정**하고 사유를 기록 — *"체중은 kg, 섭취량은 g. 규칙 테이블 단위가 체중당이라서"*

마감을 보면 **(b) 가 현실적**이다. 다만 기록 없이 두면 D-22 단일 출처 원칙이 깨진다.

### 5.2 D-43 사진 저장 자리가 없다

`Pet` 모델에 사진 칼럼이 없고, D-43 의 **관문 5단계도 아직 없다.**
아직 미구현일 뿐일 수 있으나, **프로필 사진은 D-43 에서 확정된 기능**이다.
누가 언제 붙일지 정해야 한다.

---

## 6. ⛔ 테스트가 0건이다

424줄이 들어오는데 `tests/` 에 **auth·pets 테스트가 한 건도 없다.**
게다가 CI 는 `DATABASE_URL` 이 없어 **이 모듈을 import 조차 하지 않는다** — 즉

> **이 코드는 CI 에서 단 한 줄도 실행되지 않는다.**

`ruff` 가 문법을 볼 뿐이다. 최소한 이 셋은 있어야 한다.

| 테스트 | 무엇을 막나 |
|---|---|
| 토큰 생성 → 검증 왕복 | 서명 키·알고리즘 설정 오류 |
| 만료 토큰 → 401 | `from None` 이 예외를 삼키는지 |
| 키 미설정 → `JWTKeyMissingError` | **약한 기본 키로 도는 것** |

DB 없이도 도는 테스트다 — `sqlite+pysqlite:///:memory:` 로 `Base.metadata.create_all` 하면
라우터까지 통째로 돌릴 수 있다.

---

## 7. 이서은 `lse` — 골든셋 25건

`check_goldenset.py` **ERROR 0 · WARN 17.** 형식은 깨끗하다.

### 7.1 잘한 것

- `model.revision` 을 스모크 테스트로 확인한 커밋 해시(`1cfa9a72…`)로 채웠다 —
  팀장이 미결로 남겨 둔 재현성 항목이다
- 질문이 **실제 보호자 말투**다. *"그 정도 양이면 괜찮겠죠"*, *"조금이라 괜찮겠죠"* —
  **안전한 답을 유도하는 질문**이 섞여 있는 것은 의도적으로 좋다
- `clarify` 7건 · `refused` 4건으로 04 §2.2 비율(10% · 15%)을 넘긴다

### 7.2 `must_cite` 가 14건 전부 비었다

근거 없는 정답은 채점이 안 된다. **다만 14건 모두 사실 표 888행에 근거가 실재한다** — 전수 대조했다.

```
G-001 다크초콜릿 dog 16건 · G-004 알로에 cat 3건 · G-005 초콜릿 bird 7건
G-007 고구마 dog 1건 · G-010 위확장 9건 · G-011 백합 cat 25건 · G-013 PTFE bird 11건
G-017 아보카도 bird 10건 · G-018 아스피린 dog 2건 · G-019 세제 cat 8건
G-021 방향제 bird 7건 · G-022 감기약 cat 6건 · G-025 부동액 dog 4건        → 14/14
```

**채우기만 하면 된다. 팀장이 한다.**

### 7.3 정답 자체를 확인해야 하는 3건

| 케이스 | 문제 |
|---|---|
| **G-012 달팽이약** `answered`/`CALL_NOW` | 코퍼스에 **메트알데하이드가 0건.** 잡히는 건 `살충제(유기인계·피레스로이드)` 로 **다른 물질**이다. 근거가 없어 **영원히 통과 못 한다** — 케이스를 바꾸거나 `refused/근거없음` 으로 돌린다 |
| **G-024 향초(cat)** `refused` ↔ **G-021 방향제(bird)** `answered` | 고양이 향초는 0건이라 거절이 맞고, 조류는 `S-078 향초·왁스멜트·플러그인 방향제(VOC)` 가 있어 답이 맞다. **종에 따라 갈리는 게 의도인지 확인** |
| **G-004 알로에 · G-021 방향제** | `answered` 인데 `expected_triage` 가 비었다. 등급이 없으면 **하향 금지 게이트(D-09)를 채점할 수 없다** |

### 7.4 나머지

- 커버리지 `cat 6 / bird 5` (목표 각 10) — 남은 75건에서 메운다
- `reviewed_by` **25건 전부 공란** — 04 §2.4 작성자·검수자 분리가 아직 안 됐다.
  이 문서가 그 검수 기록이므로 `reviewed_by=ohb` 로 채운다

---

## 8. 병합 순서

```
1. 이근준 — [db] extra 선언 + ImportError 조용한 실패 제거      ⛔
2. 이근준 — ChatSession 슬롯 3칼럼 제거 (또는 D-48 기록)        ⛔
3. 팀장   — 전달한 수정본 9건 반영 (JWT·lint·.gitignore)        ✅ 완료
4. 팀장   — 골든셋 must_cite 14건 + reviewed_by
5. PR #3 → main
```

**`.gitignore` 를 빼먹으면 `data/facts/` 888행이 통째로 커밋되지 않는다.**
`data/*` 가 디렉터리를 먼저 배제해서 `!data/facts/*.csv` 가 한 줄도 효력이 없었다.
`git status --short data/facts` 가 **비어 있어야** 정상이다.
