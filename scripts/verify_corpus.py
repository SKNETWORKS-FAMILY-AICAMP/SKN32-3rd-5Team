#!/usr/bin/env python
"""층 0 검증 실행 래퍼.

실제 구현은 `pettriage.tools.verify_corpus` 에 있다 — 패키지 안에 있어야
`pip install` 후에도 `pettriage-verify` 로 실행된다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pettriage.tools.verify_corpus import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
