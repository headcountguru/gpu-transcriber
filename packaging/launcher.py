"""Entry point for the PyInstaller build: behaves like ``python -m app.main``."""

import sys

from app.main import main

if __name__ == "__main__":
    sys.exit(main())
