"""Redirected output must survive non-Latin text even when Python defaults to a legacy code page."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TITLE = "千本桜 (Senbonzakura)"


def run(code: str) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING")}
    env["PYTHONIOENCODING"] = "cp1252"  # What Windows uses for redirected output by default
    return subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True)


def test_legacy_code_page_crashes_without_the_fix():
    result = run(f"print({TITLE!r})")
    assert result.returncode != 0
    assert b"UnicodeEncodeError" in result.stderr


def test_utf8_output_after_importing_stdio():
    result = run(f"import utils.stdio; print({TITLE!r}); import sys; print({TITLE!r}, file=sys.stderr)")
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout.decode("utf-8").strip() == TITLE
    assert result.stderr.decode("utf-8").strip() == TITLE


@pytest.mark.parametrize("script", ["inference.py"])
def test_entry_points_import_stdio(script):
    source = (REPO_ROOT / script).read_text(encoding="utf-8")
    assert "import utils.stdio" in source
