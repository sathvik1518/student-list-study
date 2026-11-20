import math
import shutil

import numpy as np
import pandas as pd
import pytest

from study.clean import NUMERIC, clean, normalise_section, parse_number
from study.load import assert_unchanged, load_raw, sha256_of
from tests.conftest import frame


# ---------- the raw file ----------
def test_the_raw_file_matches_its_recorded_fingerprint(raw_path):
    recorded = (raw_path.parent / "students_raw.sha256").read_text(encoding="utf-8").strip()
    assert sha256_of(raw_path) == recorded


def test_loading_reads_everything_as_text_and_keeps_NA_as_text(raw):
    assert list(raw.columns) == ["student_id", "section", "hours", "prior", "final"]
    assert all(raw[c].map(type).eq(str).all() for c in raw.columns)
    assert (raw["hours"] == "NA").any() or (raw["prior"] == "NA").any() or (raw["final"] == "NA").any() or (raw["hours"] == "n/a").any() or (raw["prior"] == "n/a").any() or (raw["final"] == "n/a").any()


def test_loading_and_cleaning_do_not_change_the_file(raw_path):
    before = raw_path.read_bytes()
    clean(load_raw(raw_path))
    assert raw_path.read_bytes() == before


def test_assert_unchanged_catches_a_change(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a\n1\n")
    sha = sha256_of(p)
    assert_unchanged(p, sha)
    p.write_text("a\n2\n")
    with pytest.raises(RuntimeError, match="changed"):
        assert_unchanged(p, sha)


# ---------- reading single values ----------
@pytest.mark.parametrize("text,value,kind", [
    ("12", 12.0, "plain"), ("7.5", 7.5, "plain"), ("-3", -3.0, "plain"), (" 8 ", 8.0, "plain"),
    ("12 hrs", 12.0, "text"), ("8h", 8.0, "text"), ("78%", 78.0, "text"), ("10 hours", 10.0, "text"),
    ("ten", 10.0, "word"), ("Twelve", 12.0, "word"), ("zero", 0.0, "word"),
])
def test_parse_number_reads_values(text, value, kind):
    assert parse_number(text) == (value, kind)


@pytest.mark.parametrize("text", ["", "NA", "n/a", "N/A", "nan", " ", "-", "None"])
def test_parse_number_gaps(text):
    v, kind = parse_number(text)
    assert math.isnan(v) and kind == "gap"


@pytest.mark.parametrize("text", ["lots", "12 apples", "1,5", "1e3x", "twelve hrs"])
def test_parse_number_unreadable_is_a_gap_not_a_guess(text):
    v, kind = parse_number(text)
    assert math.isnan(v) and kind == "unreadable"


@pytest.mark.parametrize("text,expected", [
    ("A", "A"), ("a", "A"), (" B ", "B"), ("Sec C", "C"), ("section b", "B"), ("B-", "B"), ("SECTION A", "A"),
    ("D", "Unknown"), ("", "Unknown"), ("AB", "Unknown"), ("1", "Unknown"),
])
def test_normalise_section(text, expected):
    assert normalise_section(text) == expected


# ---------- cleaning small hand-made tables ----------
def test_exact_repeat_is_removed_and_counted():
    res = clean(frame([("S1", "A", "5", "60", "70"), ("S1", "A", "5", "60", "70"), ("S2", "B", "6", "50", "60")]))
    assert list(res.data["student_id"]) == ["S1", "S2"]
    assert res.count("repeat rows removed (exact copy)") == 1
    assert res.count("repeat rows removed (different numbers)") == 0
    assert res.removed["reason"].str.contains("exact repeat").sum() == 1


def test_repeat_with_different_numbers_keeps_the_first_and_is_counted_separately():
    res = clean(frame([("S1", "A", "5", "60", "70"), ("S1", "A", "5", "60", "79")]))
    assert res.data.loc[0, "final"] == 70
    assert res.count("repeat rows removed (different numbers)") == 1
    assert res.count("repeat rows removed (exact copy)") == 0
    assert "DIFFERENT" in res.removed.loc[0, "reason"]


def test_impossible_values_become_gaps_but_the_row_stays():
    res = clean(frame([("S1", "A", "940", "60", "70"), ("S2", "A", "-3", "60", "70"), ("S3", "A", "5", "-5", "70"), ("S4", "A", "5", "60", "130")]))
    assert list(res.data["student_id"]) == ["S1", "S2", "S3"]          # S4 lost its final -> out of the analysis
    assert res.data["hours"].isna().tolist() == [True, True, False]
    assert res.data.loc[2, "prior"] != res.data.loc[2, "prior"]        # NaN
    assert res.count("hours: impossible value") == 2 and res.count("prior: impossible value") == 1 and res.count("final: impossible value") == 1
    assert res.removed["student_id"].tolist() == ["S4"] and res.removed.loc[0, "reason"] == "no usable final score"


def test_the_naive_version_keeps_impossible_values():
    res = clean(frame([("S1", "A", "940", "60", "70")]), blank_impossible=False)
    assert res.data.loc[0, "hours"] == 940


def test_no_final_means_removed_from_the_analysis_with_a_reason():
    res = clean(frame([("S1", "A", "5", "60", ""), ("S2", "A", "5", "60", "NA"), ("S3", "A", "5", "60", "55")]))
    assert list(res.data["student_id"]) == ["S3"]
    assert set(res.removed["student_id"]) == {"S1", "S2"}


def test_section_spelling_is_fixed_and_counted():
    res = clean(frame([("S1", "Sec b", "5", "60", "70"), ("S2", "A", "5", "60", "70"), ("S3", "??", "5", "60", "70")]))
    assert res.data["section"].tolist() == ["B", "A", "Unknown"]
    assert res.count("section spelling fixed") == 2          # 'Sec b' -> B and '??' -> Unknown both changed
    assert res.count("section unknown") == 1


def test_text_numbers_are_read_and_counted():
    res = clean(frame([("S1", "A", "12 hrs", "60", "78%"), ("S2", "A", "ten", "60", "70")]))
    assert res.data["hours"].tolist() == [12.0, 10.0] and res.data["final"].tolist() == [78.0, 70.0]
    assert res.count("hours: text") == 1 and res.count("hours: word") == 1 and res.count("final: text") == 1


def test_whitespace_around_ids_is_ignored_when_finding_repeats():
    res = clean(frame([("S1", "A", "5", "60", "70"), (" S1 ", "A", "5", "60", "70")]))
    assert len(res.data) == 1


def test_the_input_table_is_not_modified():
    df = frame([("S1", " Sec b", "12 hrs", "60", "70")])
    copy = df.copy(deep=True)
    clean(df)
    pd.testing.assert_frame_equal(df, copy)


# ---------- cleaning the real raw file ----------
def test_every_raw_row_is_either_used_or_removed_with_a_reason(raw):
    res = clean(raw)
    assert res.n_raw == len(raw) == len(res.data) + len(res.removed)
    assert res.removed["reason"].str.len().gt(10).all()
    assert not res.removed["reason"].isna().any()


def test_cleaned_data_is_sound(raw):
    d = clean(raw).data
    assert d["student_id"].is_unique
    assert d["final"].notna().all()
    for c in NUMERIC:
        assert d[c].dropna().between(0, 100).all()
    assert set(d["section"]) <= {"A", "B", "C", "Unknown"}


def test_the_counts_in_the_log_match_a_separate_count_on_the_raw_file(raw):
    res = clean(raw)
    ids = raw["student_id"]
    assert res.count("repeat rows removed (exact copy)") + res.count("repeat rows removed (different numbers)") == int(ids.duplicated().sum())
    nums = {c: pd.to_numeric(raw[c], errors="coerce") for c in NUMERIC}
    assert res.count("hours: impossible value") == int(((nums["hours"] < 0) | (nums["hours"] > 100)).sum()) - int(
        (((nums["hours"] < 0) | (nums["hours"] > 100)) & ids.duplicated()).sum())
    assert (raw["hours"].str.strip() == "940").sum() == 1                      # the 940 row is really there...
    assert 940 not in clean(raw).data["hours"].dropna().tolist()               # ...and really gone
    assert 940 in clean(raw, blank_impossible=False).data["hours"].tolist()    # ...unless we keep it on purpose


def test_cleaning_twice_gives_the_same_answer(raw):
    a, b = clean(raw), clean(raw)
    pd.testing.assert_frame_equal(a.data, b.data)
    pd.testing.assert_frame_equal(a.removed, b.removed)


def test_cleaning_does_not_depend_on_row_order(raw):
    a = clean(raw).data.sort_values("student_id").reset_index(drop=True)
    shuffled = raw.sample(frac=1, random_state=3).reset_index(drop=True)
    b = clean(shuffled)
    # the repeated student with DIFFERENT numbers can keep a different row when the order changes, so compare everybody else
    clash = set(b.removed.loc[b.removed["reason"].str.contains("DIFFERENT"), "student_id"])
    a2, b2 = a[~a["student_id"].isin(clash)], b.data[~b.data["student_id"].isin(clash)].sort_values("student_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(a2.reset_index(drop=True), b2)
