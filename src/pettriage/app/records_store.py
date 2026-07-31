"""다이어리 기록 저장소 — 데모 단계의 자리.

설계 근거: docs/02_시스템-아키텍처.md §12.3 · docs/06 D-18 · D-36

    ⚠️ **인증이 없다.** 이 골격에는 사용자 인증·인가가 없으므로
    `pet_id` 를 아는 사람은 누구나 그 기록을 읽는다.

    과제 산출물에 인증은 포함되지 않지만, **없다는 사실을 코드에 적어 둔다** —
    "개인정보 리스크가 낮은 도메인"(D-01)이라는 판단이
    "접근 통제가 필요 없다"로 번지면 안 된다.

    실사용으로 나가려면 여기에 소유자 확인이 붙어야 하고,
    그 전까지 **가상 프로필만** 넣는다 (D-18).

모듈 전역이 아니라 주입 가능한 객체로 둔다. 전역이면 앱 인스턴스가 달라도
같은 데이터를 보게 되어 테스트가 서로를 오염시킨다.
"""

from __future__ import annotations

import threading
import uuid
from typing import Any

#: 조류가 아닌 종에서는 버리는 필드 (02 §12).
BIRD_ONLY_FIELDS = ("droppings",)


class RecordStore:
    """메모리 기록 저장소. WS1이 pgvector/Chroma 적재로 교체한다 (02 §11)."""

    def __init__(self) -> None:
        self._data: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def add(self, payload: dict[str, Any]) -> str:
        record_id = uuid.uuid4().hex[:12]
        row = dict(payload)
        # 종이 맞을 때만 조류 전용 필드를 보관한다 — 최소 수집 (D-36)
        if row.get("species") != "bird":
            for f in BIRD_ONLY_FIELDS:
                row.pop(f, None)
        row["record_id"] = record_id
        with self._lock:
            self._data.setdefault(row["pet_id"], []).append(row)
        return record_id

    def timeline(
        self, pet_id: str, period_from: str = "", period_to: str = ""
    ) -> list[dict[str, Any]]:
        """기간 필터를 적용한 기록. 날짜는 ISO 8601 문자열 비교로 자른다.

        받기만 하고 쓰지 않으면 화면의 기간 선택이 거짓말이 된다.
        """
        with self._lock:
            rows = list(self._data.get(pet_id, []))
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
