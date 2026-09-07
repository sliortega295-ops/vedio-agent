import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE / "third_party"))
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "tests"))
