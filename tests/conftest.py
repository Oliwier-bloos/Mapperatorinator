import sys
from pathlib import Path

# The project imports its modules from the repository root (e.g. `osuT5.osuT5...`)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
