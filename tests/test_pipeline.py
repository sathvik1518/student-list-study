import json
import shutil

import pandas as pd
import pytest

from study import pipeline, report
from study.load import sha256_of


@pytest.fixture(scope="module")
def run_once(tmp_path_factory, raw_path):
    out = tmp_path_factory.mktemp("out")
    before = raw_path.read_bytes()
    summary = pipeline.run(raw_path, out)
    return summary, out, before


def test_the_raw_file_is_byte_for_byte_the_same_after_a_full_run(run_once, raw_path):
    _, _, before = run_once
    assert raw_path.read_bytes() == before


def test_all_outputs_exist_and_nothing_is_written_next_to_the_raw_file(run_once, raw_path):
    _, out, _ = run_once
    for name in ["report.md", "section_table.csv", "removed_rows.csv", "cleaning_log.csv", "clean_students.csv", "train_filled.csv", "test_filled.csv", "tests_run.csv", "summary.json"]:
        assert (out / name).stat().st_size > 0, name
    for png in ["hours_vs_final.png", "section_means.png", "gradient_descent_cost.png"]:
        data = (out / "charts" / png).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) > 5000
    assert sorted(p.name for p in raw_path.parent.iterdir()) == ["students_raw.csv", "students_raw.sha256"]


def test_row_accounting_adds_up(run_once):
    s, out, _ = run_once
    removed = pd.read_csv(out / "removed_rows.csv")
    used = pd.read_csv(out / "clean_students.csv")
    assert s["raw_rows"] == s["analysis_rows"] + s["removed_rows"] == len(used) + len(removed)


def test_the_number_of_tests_in_the_report_equals_the_number_really_run(run_once):
    s, out, _ = run_once
    tests = pd.read_csv(out / "tests_run.csv")
    assert s["tests_run"] == len(tests) == 4
    assert f"**{len(tests)} tests were run**" in (out / "report.md").read_text(encoding="utf-8")
    assert (tests["p_holm"] >= tests["p"] - 1e-12).all()


def test_the_train_and_test_files_are_disjoint_and_have_no_gaps_left(run_once):
    _, out, _ = run_once
    tr, te = pd.read_csv(out / "train_filled.csv"), pd.read_csv(out / "test_filled.csv")
    assert set(tr["student_id"]).isdisjoint(te["student_id"])
    assert tr[["hours", "prior", "final"]].notna().all().all() and te[["hours", "prior", "final"]].notna().all().all()
    # every filled value in BOTH files is the TRAIN median
    s = json.loads((out / "summary.json").read_text(encoding="utf-8"))["fit"]["train_medians"]
    for df in (tr, te):
        for col in ("hours", "prior"):
            assert df.loc[df[col + "_was_gap"], col].eq(s[col]).all()
    assert tr["hours_was_gap"].sum() + te["hours_was_gap"].sum() > 0
    # independent check: the fill value is the median of the TRAIN rows that had a real value, and it is NOT the median of all rows
    all_rows = pd.concat([tr, te])
    for col in ("hours", "prior"):
        assert tr.loc[~tr[col + "_was_gap"], col].median() == s[col]
    assert any(all_rows.loc[~all_rows[col + "_was_gap"], col].median() != s[col] for col in ("hours", "prior"))


def test_gradient_descent_in_the_run_agrees_with_least_squares(run_once):
    fit = run_once[0]["fit"]
    assert fit["max_difference"] < 1e-6
    assert fit["final_cost"] < fit["first_cost"] / 10


def test_the_model_beats_guessing_the_average_on_the_test_students(run_once):
    fit = run_once[0]["fit"]
    assert fit["test_rmse"] < fit["baseline_rmse"]


def test_the_report_has_every_required_part(run_once):
    text = (run_once[1] / "report.md").read_text(encoding="utf-8")
    for must in ["The answer (one page)", "Average final score by section", "Pearson", "Spearman", "Rows that were removed", "The 940-hour row", "Holm", "Train / test split",
                 "What I did not check", "never changed", "section_means.png", "hours_vs_final.png", "gradient_descent_cost.png"]:
        assert must in text, must
    assert text.count("\n- ") >= 15


def test_a_second_run_gives_identical_numbers(run_once, raw_path, tmp_path):
    s1 = run_once[0]
    s2 = pipeline.run(raw_path, tmp_path, make_charts=False)
    for k in ["raw_rows", "analysis_rows", "removed_rows", "tests_run"]:
        assert s1[k] == s2[k]
    assert s1["corr"] == s2["corr"] and s1["fit"]["test_rmse"] == s2["fit"]["test_rmse"]


def test_a_run_that_touches_the_raw_file_is_stopped(raw_path, tmp_path, monkeypatch):
    copy = tmp_path / "students_raw.csv"
    shutil.copy(raw_path, copy)

    def naughty(*a, **k):
        with open(copy, "a") as f:
            f.write("S999,A,1,1,1\n")
        return "x"

    monkeypatch.setattr(pipeline, "build_report", naughty)
    with pytest.raises(RuntimeError, match="changed"):
        pipeline.run(copy, tmp_path / "out", make_charts=False)


def test_a_different_seed_changes_the_split_but_not_the_cleaning(raw_path, tmp_path):
    a = pipeline.run(raw_path, tmp_path / "a", seed=1, make_charts=False)
    b = pipeline.run(raw_path, tmp_path / "b", seed=2, make_charts=False)
    assert a["analysis_rows"] == b["analysis_rows"] and a["fit"]["test_rmse"] != b["fit"]["test_rmse"]


# ---------- the wording follows the numbers ----------
def _inputs(p_pearson, low, p_anova, holm_pearson=None, holm_anova=None):
    import pandas as pd
    table = pd.DataFrame([{"section": "A", "n": 10, "mean_final": 60.0, "sd": 5.0, "ci_low": 57.0, "ci_high": 63.0}])
    pair = {"diff": 1.0, "ci_low": -2.0, "ci_high": 4.0}
    anova = {"p": p_anova}
    corr = {"pearson": 0.5, "pearson_low": low, "pearson_high": 0.7, "spearman": 0.5}
    adjusted = {"slope": 1.5, "low": 1.0, "high": 2.0}
    hp = {"pearson": 0.2}
    tests = pd.DataFrame([{"test": "One-way ANOVA, final by section", "p": p_anova, "p_holm": p_anova if holm_anova is None else holm_anova},
                          {"test": "Pearson correlation, hours vs final", "p": p_pearson, "p_holm": p_pearson if holm_pearson is None else holm_pearson}])
    return table, pair, anova, corr, adjusted, hp, tests


def test_answer_says_yes_only_when_the_numbers_say_so():
    yes = " ".join(report.answer_lines(*[_inputs(0.001, 0.4, 0.5)[i] for i in (0, 1, 2, 3, 4, 5, 6)]))
    assert "Hours: yes" in yes and "no clear sign" in yes
    no = " ".join(report.answer_lines(*_inputs(0.4, -0.1, 0.5)))
    assert "Hours: no clear link" in no
    sec = " ".join(report.answer_lines(*_inputs(0.001, 0.4, 0.01)))
    assert "it seems to matter" in sec and "no clear sign" not in sec


def test_answer_always_warns_that_this_is_not_proof_of_cause():
    for args in (_inputs(0.001, 0.4, 0.5), _inputs(0.4, -0.1, 0.01)):
        assert "not proof" in " ".join(report.answer_lines(*args))


def test_the_standardiser_numbers_come_from_the_train_rows_only(run_once):
    s, out, _ = run_once
    tr = pd.read_csv(out / "train_filled.csv")
    assert s["fit"]["prior_mean_train"] == pytest.approx(tr["prior"].mean())
    assert s["fit"]["prior_sd_train"] == pytest.approx(tr["prior"].std(ddof=0))


@pytest.mark.parametrize("p_pearson,low,expect_yes", [(0.001, 0.4, True), (0.001, -0.05, False), (0.2, 0.3, False), (0.049, 0.01, True), (0.05, 0.3, False)])
def test_hours_answer_needs_both_a_small_corrected_p_and_an_interval_above_zero(p_pearson, low, expect_yes):
    text = " ".join(report.answer_lines(*_inputs(p_pearson, low, 0.5)))
    assert ("Hours: yes" in text) is expect_yes


def test_the_answer_uses_the_corrected_p_not_the_raw_one():
    # raw p looks significant, corrected p does not
    text = " ".join(report.answer_lines(*_inputs(0.03, 0.4, 0.03, holm_pearson=0.12, holm_anova=0.12)))
    assert "Hours: no clear link" in text and "no clear sign" in text
