import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402


@pytest.fixture(scope="session")
def raw_path():
    return ROOT / "data" / "students_raw.csv"


@pytest.fixture(scope="session")
def raw(raw_path):
    from study.load import load_raw
    return load_raw(raw_path)


def frame(rows):
    """rows: list of (student_id, section, hours, prior, final) as TEXT, exactly like the raw file."""
    return pd.DataFrame(rows, columns=["student_id", "section", "hours", "prior", "final"]).astype(str)
