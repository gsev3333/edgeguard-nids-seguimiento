"""Compatibility entry point for the canonical EdgeGuard training script."""
from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.model.train_real import main


if __name__ == "__main__":
    main()
