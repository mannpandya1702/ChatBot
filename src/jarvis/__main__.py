"""``python -m jarvis`` entrypoint."""

from __future__ import annotations

import sys

from jarvis.main import main

if __name__ == "__main__":
    sys.exit(main())
