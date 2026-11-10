"""Enable `python -m bot` execution."""
from __future__ import annotations

import sys

from bot.cli import main

if __name__ == "__main__":
    sys.exit(main())
