"""Cleaning in memory. The raw file is never touched. Every change is counted, and every removed row is listed with a reason.

Rules (written down so they can be argued with):
  1. A student id that appears twice is a repeat. Keep the first row, remove the others.
     (If the repeat has DIFFERENT numbers we still keep the first, but we count that case separately: we cannot know which one is right.)
  2. Numbers written as text are read: "12 hrs" -> 12, "78%" -> 78, "ten" -> 10.  Anything else unreadable becomes a gap.
  3. "NA", "n/a", "" are gaps.
  4. Section is spelled many ways ("Sec b", " B-") -> A, B, C. Anything else -> "Unknown".
  5. Impossible values become gaps (the row stays): hours outside 0-100, prior or final outside 0-100.
  6. A student with no final score cannot be used to study the final score. That row is removed from the analysis.
"""
import math
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NUMBER_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
GAP_TEXT = {"", "na", "n/a", "nan", "none", "null", "-", "--"}
PLAIN = re.compile(r"^[-+]?\d+(\.\d+)?$")
WITH_UNIT = re.compile(r"^([-+]?\d+(\.\d+)?)\s*(hrs?|h|hours?|%|pts?|points?)$")

LIMITS = {"hours": (0.0, 100.0), "prior": (0.0, 100.0), "final": (0.0, 100.0)}
NUMERIC = ["hours", "prior", "final"]


def parse_number(text: str) -> tuple[float, str]:
    """Return (value, kind). kind is one of: plain, text, word, gap, unreadable."""
    t = str(text).strip().lower()
    if t in GAP_TEXT:
        return math.nan, "gap"
    if PLAIN.match(t):
        return float(t), "plain"
    m = WITH_UNIT.match(t)
    if m:
        return float(m.group(1)), "text"
    if t in NUMBER_WORDS:
        return float(NUMBER_WORDS[t]), "word"
    return math.nan, "unreadable"


def normalise_section(text: str) -> str:
    t = re.sub(r"(section|sec)", "", str(text).strip().lower())
    t = re.sub(r"[^a-z]", "", t)
    return t.upper() if t in {"a", "b", "c"} else "Unknown"


@dataclass
class CleanResult:
    data: pd.DataFrame                      # the rows used for the analysis
    removed: pd.DataFrame                   # every removed row, with the reason
    log: list[dict] = field(default_factory=list)
    n_raw: int = 0

    def count(self, action: str) -> int:
        return int(sum(item["count"] for item in self.log if item["action"] == action))


def clean(raw: pd.DataFrame, blank_impossible: bool = True) -> CleanResult:
    """blank_impossible=False is the "naive" version used only to show what happens if you do NOT remove the 940-hour row."""
    log: list[dict] = []

    def note(action: str, count: int, why: str, ids: list[str] | None = None) -> None:
        log.append({"action": action, "count": int(count), "why": why, "examples": ", ".join((ids or [])[:5])})

    df = raw.copy()
    n_raw = len(df)
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip()

    # 1. repeats
    removed_parts = []
    dup_mask = df.duplicated(subset="student_id", keep="first")
    first = df[~dup_mask].set_index("student_id")
    exact, differ = [], []
    for idx in df[dup_mask].index:
        row = df.loc[idx]
        (exact if first.loc[row["student_id"]].astype(str).tolist() == row[["section", "hours", "prior", "final"]].astype(str).tolist() else differ).append(idx)
    if exact:
        removed_parts.append(df.loc[exact].assign(reason="exact repeat of an earlier row"))
    if differ:
        removed_parts.append(df.loc[differ].assign(reason="same student id with DIFFERENT numbers (kept the first row, cannot know which is right)"))
    note("repeat rows removed (exact copy)", len(exact), "same student id and same numbers", df.loc[exact, "student_id"].tolist())
    note("repeat rows removed (different numbers)", len(differ), "same student id, different numbers; first row kept", df.loc[differ, "student_id"].tolist())
    df = df[~dup_mask].copy()

    # 2/3. numbers
    for col in NUMERIC:
        values, kinds = zip(*[parse_number(v) for v in df[col]])
        df[col + "_kind"] = kinds
        df[col] = values
        for kind, why in [("text", "number written with a unit, like '12 hrs' or '78%'"), ("word", "number written as a word, like 'ten'"),
                          ("gap", "empty / NA / n/a: left as a gap"), ("unreadable", "could not be read: made a gap")]:
            n = int((df[col + "_kind"] == kind).sum())
            if n:
                note(f"{col}: {kind}", n, why, df.loc[df[col + "_kind"] == kind, "student_id"].tolist())

    # 4. section
    before = df["section"].copy()
    df["section"] = before.map(normalise_section)
    changed = int((before != df["section"]).sum())
    note("section spelling fixed", changed, "e.g. 'Sec b', ' B-' -> B", df.loc[before != df["section"], "student_id"].tolist())
    note("section unknown", int((df["section"] == "Unknown").sum()), "could not tell the section", df.loc[df["section"] == "Unknown", "student_id"].tolist())

    # 5. impossible values
    for col, (lo, hi) in LIMITS.items():
        bad = (df[col] < lo) | (df[col] > hi)
        if bad.any():
            note(f"{col}: impossible value", int(bad.sum()), f"outside {lo:g}-{hi:g}" + ("" if blank_impossible else " (KEPT in the naive version)"),
                 [f"{i}={v:g}" for i, v in zip(df.loc[bad, 'student_id'], df.loc[bad, col])])
            if blank_impossible:
                df.loc[bad, col] = np.nan

    # 6. no final score -> out of the analysis
    nofinal = df["final"].isna()
    if nofinal.any():
        removed_parts.append(df[nofinal][["student_id", "section", "hours", "prior", "final"]].astype(str).assign(reason="no usable final score"))
    note("rows removed: no usable final score", int(nofinal.sum()), "cannot study the final score without it", df.loc[nofinal, "student_id"].tolist())
    df = df[~nofinal].copy()

    df = df[["student_id", "section", "hours", "prior", "final"]].reset_index(drop=True)
    removed = pd.concat(removed_parts) if removed_parts else pd.DataFrame(columns=["student_id", "section", "hours", "prior", "final", "reason"])
    removed = removed[["student_id", "section", "hours", "prior", "final", "reason"]].reset_index(drop=True)
    return CleanResult(data=df, removed=removed, log=log, n_raw=n_raw)
