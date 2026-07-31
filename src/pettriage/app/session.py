"""되묻기 세션 — 휘발성 상태.

설계 근거: docs/05_설계원칙-코드와LLM의분업.md §3

    이 프로젝트에서 "기억"으로 불릴 수 있는 것이 둘인데 완전히 다르게 처리한다.

      · 되묻기 세션 상태  → **여기.** LangGraph State에 대응. 휘발성.
      · 반려동물 일일 기록 → 조각 3이 아니라 **조각 4(RAG)**. 벡터DB.

    후자를 "장기 기억"이라 부르면 설계가 흐려진다. 그래서 이 저장소는
    질의 슬롯만 담고, 기록은 절대 들어오지 않는다.

메모리 구현이다. 프로세스가 죽으면 사라지는 것이 **의도**다 —
되묻기 슬롯(체중·섭취량)은 보관할 이유가 없다 (D-36 최소 수집).
다중 워커로 띄울 때만 Redis 등으로 교체한다 (02 §13).

라우터가 동기 함수라 FastAPI 스레드풀에서 **진짜 병렬로** 실행된다.
그래서 저장소 조작에 락을 건다.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field

from .contracts import AskRequest

#: 되묻기가 이 시간 안에 안 끝나면 세션을 버린다. 슬롯을 오래 들고 있지 않는다.
SESSION_TTL_SEC = 30 * 60
MAX_SESSIONS = 1000

#: 되묻기로 채워질 수 있는 슬롯. `merge` 가 이 목록만 옮긴다.
SLOTS = ("species", "pet_id", "weight_kg", "amount_g")


@dataclass
class Session:
    session_id: str
    species: str | None = None
    pet_id: str | None = None
    weight_kg: float | None = None
    amount_g: float | None = None
    clarify_turns: int = 0
    created_at: float = field(default_factory=time.monotonic)
    touched_at: float = field(default_factory=time.monotonic)

    def merge(self, req: AskRequest) -> bool:
        """새 요청에서 채워진 슬롯만 받아 덮는다. None은 기존 값을 지우지 않는다.

        Returns:
            **비어 있던 슬롯이 새로 채워졌는가** — 되묻기에 진전이 있었는지.
            진전이 있으면 되묻기 카운터를 되돌린다. 그러지 않으면 협조적인
            사용자가 슬롯을 하나씩 채우다가 상한에 걸려 거절된다.
        """
        progressed = False
        for f in SLOTS:
            v = getattr(req, f)
            if v is None:
                continue
            if getattr(self, f) is None:
                progressed = True
            setattr(self, f, v)
        self.touched_at = time.monotonic()
        return progressed


class SessionStore:
    """프로세스 메모리 세션 저장소. 스레드 안전."""

    def __init__(self, ttl: float = SESSION_TTL_SEC, max_size: int = MAX_SESSIONS) -> None:
        self._data: dict[str, Session] = {}
        self._ttl = ttl
        self._max = max_size
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str | None) -> Session:
        with self._lock:
            self._evict()
            if session_id and session_id in self._data:
                s = self._data[session_id]
                s.touched_at = time.monotonic()  # 대화 중 TTL로 사라지지 않게
                return s
            # 클라이언트가 보낸 미지의 id는 신뢰하지 않고 새로 발급한다.
            sid = uuid.uuid4().hex
            self._data[sid] = Session(session_id=sid)
            self._evict()  # 삽입 뒤에도 상한을 지킨다
            return self._data[sid]

    def _evict(self) -> None:
        """만료·초과 세션 제거. 락을 쥔 상태에서만 부른다."""
        now = time.monotonic()
        for sid in [s for s, v in list(self._data.items()) if now - v.touched_at > self._ttl]:
            self._data.pop(sid, None)
        if len(self._data) > self._max:
            oldest = sorted(self._data.items(), key=lambda kv: kv[1].touched_at)
            for sid, _ in oldest[: len(self._data) - self._max]:
                self._data.pop(sid, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
