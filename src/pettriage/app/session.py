"""되묻기 세션 — 휘발성 상태.

설계 근거: docs/05_설계원칙-코드와LLM의분업.md §3

    이 프로젝트에서 "기억"으로 불릴 수 있는 것이 둘인데 완전히 다르게 처리한다.

      · 되묻기 세션 상태  → **여기.** LangGraph State에 대응. 휘발성.
      · 반려동물 일일 기록 → 조각 3이 아니라 **조각 4(RAG)**. 벡터DB.

    후자를 "장기 기억"이라 부르면 설계가 흐려진다. 그래서 이 저장소는
    질의 슬롯만 담고, 기록은 절대 들어오지 않는다.

메모리 구현이다. 프로세스가 죽으면 사라지는 것이 **의도**다 —
되묻기 슬롯(체중·섭취량)은 보관할 이유가 없다 (D-36 최소 수집).
다중 워커로 띄울 때만 Redis 등으로 교체한다.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .contracts import AskRequest

#: 되묻기가 이 시간 안에 안 끝나면 세션을 버린다. 슬롯을 오래 들고 있지 않는다.
SESSION_TTL_SEC = 30 * 60
MAX_SESSIONS = 1000


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

    def merge(self, req: AskRequest) -> None:
        """새 요청에서 채워진 슬롯만 받아 덮는다. None은 기존 값을 지우지 않는다."""
        for f in ("species", "pet_id", "weight_kg", "amount_g"):
            v = getattr(req, f)
            if v is not None:
                setattr(self, f, v)
        self.touched_at = time.monotonic()


class SessionStore:
    """프로세스 메모리 세션 저장소."""

    def __init__(self, ttl: float = SESSION_TTL_SEC, max_size: int = MAX_SESSIONS) -> None:
        self._data: dict[str, Session] = {}
        self._ttl = ttl
        self._max = max_size

    def get_or_create(self, session_id: str | None) -> Session:
        self._evict()
        if session_id and session_id in self._data:
            return self._data[session_id]
        # 클라이언트가 보낸 미지의 id는 신뢰하지 않고 새로 발급한다.
        sid = uuid.uuid4().hex
        self._data[sid] = Session(session_id=sid)
        return self._data[sid]

    def _evict(self) -> None:
        now = time.monotonic()
        for sid in [s for s, v in self._data.items() if now - v.touched_at > self._ttl]:
            del self._data[sid]
        if len(self._data) > self._max:
            oldest = sorted(self._data.items(), key=lambda kv: kv[1].touched_at)
            for sid, _ in oldest[: len(self._data) - self._max]:
                del self._data[sid]

    def __len__(self) -> int:
        return len(self._data)
