"""Allow running `python -m svgder`."""

import sys

from svgder.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
