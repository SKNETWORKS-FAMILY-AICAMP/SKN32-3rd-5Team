"""다이어리 기록 저장소 — 데모 단계의 자리.

설계 근거: docs/02_시스템-아키텍처.md §12.3 · docs/06 D-18 · D-36

    소유자 확인 — 2026-08-02에 붙었다
    -------------------------------
    예전에는 `pet_id` 만 열쇠였다. 그래서 **`pet_id` 를 아는 사람은 누구나
    그 기록을 읽었고**, 인증 헤더 없이 `GET /api/report?pet_id=...` 가 200 을 냈다.

        {'note': '어제 구토, 주인 전화 010-1234-5678', 'symptoms': ['구토'], …}

    이 주석에는 원래 *"인증이 없다"* 고만 적혀 있었다. 그 뒤 `auth`·`pets` 가
    들어오면서 다른 라우터는 전부 `user_id` 로 잘렸는데 **여기만 남았다** —
    *"없다고 적어 두었다"* 가 *"고쳐도 되는 것"* 을 가려 버린 셈이다.

    지금은 열쇠가 `(owner_id, pet_id)` 다. `owner_id` 는 `deps.get_owner_id` 가 준다.
    DB 구성에서는 인증된 사용자 id, DB 없는 데모 구성에서는 단일 소유자다
    (그 구성에는 사용자 개념 자체가 없으므로 넘나들 상대도 없다).

    ⚠️ 여전히 **가상 프로필만** 넣는다 (D-18). 소유자 확인이 붙었다고
    실제 개인정보를 넣어도 된다는 뜻이 아니다.

모듈 전역이 아니라 주입 가능한 객체로 둔다. 전역이면 앱 인스턴스가 달라도
같은 데이터를 보게 되어 테스트가 서로를 오염시킨다.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

#: 조류가 아닌 종에서는 버리는 필드 (02 §12).
BIRD_ONLY_FIELDS = ("droppings",)

#: 메모리 저장소 전체 상한. 넘으면 오래된 것부터 버린다.
MAX_RECORDS = 10_000


class RecordStore:
    """메모리 기록 저장소. **일일 기록은 아직 어디에도 영구 저장되지 않는다.**

    WS1 이 **Chroma `internal` 컬렉션** 적재로 교체한다 (02 §3 · §11.1).
    ⚠️ 예전 주석은 *"pgvector/Chroma"* 였으나 **pgvector 는 D-44 로 걷어냈다.**

    **MySQL 이 아니다.** 이 프로젝트의 저장소는 셋이고 담는 것이 다르다 (02 §11.1).

        MySQL            계정 · 반려동물 프로필 (누가 · 무엇을 기르나)   ← D-48
        Chroma external  공적 지식 888청크                              ← 완료
        Chroma internal  일일 기록                                      ← 여기가 갈 자리

    일일 기록을 MySQL 에 넣으면 **검색 대상이 아니게 된다** — 05 §3 이
    *"일일 기록은 조각 3(기억)이 아니라 조각 4(RAG)"* 로 못박은 이유다.
    """

    def __init__(self, max_records: int = MAX_RECORDS) -> None:
        self._data: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self._lock = threading.Lock()
        self._max = max_records

    def add(self, owner_id: str, payload: dict[str, Any]) -> str:
        record_id = uuid.uuid4().hex[:12]
        row = dict(payload)
        # 종이 맞을 때만 조류 전용 필드를 보관한다 — 최소 수집 (D-36)
        if row.get("species") != "bird":
            for f in BIRD_ONLY_FIELDS:
                row.pop(f, None)
        row["record_id"] = record_id
        with self._lock:
            self._data.setdefault((owner_id, row["pet_id"]), []).append(row)
            self._evict()
        return record_id

    def _evict(self) -> None:
        """전체 상한을 지킨다. 락을 쥔 상태에서만 부른다.

        `SessionStore` 에는 TTL·상한이 있는데 여기에는 없었다. 메모리 저장소가
        무한히 자라면 **기록이 많은 사용자가 프로세스를 죽인다.**
        오래된 것부터 버린다 — 영구 저장은 WS1 의 Chroma `internal` 이 맡는다.
        """
        total = sum(len(v) for v in self._data.values())
        if total <= self._max:
            return
        drop = total - self._max
        for key in list(self._data):
            if drop <= 0:
                break
            rows = self._data[key]
            cut = min(drop, len(rows))
            del rows[:cut]
            drop -= cut
            if not rows:
                del self._data[key]

    def timeline(
        self, owner_id: str, pet_id: str, period_from: str = "", period_to: str = ""
    ) -> list[dict[str, Any]]:
        """기간 필터를 적용한 기록. 날짜는 ISO 8601 문자열 비교로 자른다.

        받기만 하고 쓰지 않으면 화면의 기간 선택이 거짓말이 된다.
        """
        with self._lock:
            rows = list(self._data.get((owner_id, pet_id), []))
        if period_from:
            rows = [r for r in rows if str(r.get("recorded_at", "")) >= period_from]
        if period_to:
            # 종료일 당일을 포함시킨다 (`2026-07-31` < `2026-07-31T09:00`)
            rows = [r for r in rows if str(r.get("recorded_at", ""))[:10] <= period_to[:10]]
        return sorted(rows, key=lambda r: str(r.get("recorded_at", "")))

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._data.values())
