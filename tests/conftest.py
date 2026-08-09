"""Make `import garudata` work from a plain checkout (no install needed)."""
import importlib.util
import sys
from pathlib import Path

if importlib.util.find_spec("garudata") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
