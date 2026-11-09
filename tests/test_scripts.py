"""Every command-line script and the dashboard at least compile (scripts are not imported by other tests)."""
import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "scripts").rglob("*.py")) + [ROOT / "app" / "streamlit_app.py"]


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_compiles(path):
    py_compile.compile(str(path), doraise=True)
