"""Load the raw file WITHOUT changing it, and prove afterwards that it was not changed."""
import hashlib
from pathlib import Path

import pandas as pd


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_raw(path: Path) -> pd.DataFrame:
    """Everything is read as TEXT. Nothing is guessed, converted or filled here, and "NA" stays "NA"."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def assert_unchanged(path: Path, expected_sha: str) -> None:
    now = sha256_of(path)
    if now != expected_sha:
        raise RuntimeError(f"The raw file changed! It was {expected_sha[:12]}... and is now {now[:12]}.... Restore it with:  git checkout data/students_raw.csv")
