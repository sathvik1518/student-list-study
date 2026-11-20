"""The whole study as one function, so the script and the tests run the very same code."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import charts, model, stats
from .clean import clean
from .load import assert_unchanged, load_raw, sha256_of
from .report import build_report

GD_RATES = (0.01, 0.1, 0.5)
MAIN_RATE = 0.1
GD_STEPS = 500
PRE_CHOSEN_PAIR = ("A", "B")   # chosen before looking at any numbers


def run(raw_path: Path, out_dir: Path, seed: int = 51, make_charts: bool = True) -> dict:
    raw_path, out_dir = Path(raw_path), Path(out_dir)
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)
    sha_before = sha256_of(raw_path)

    raw = load_raw(raw_path)
    result = clean(raw)
    naive = clean(raw, blank_impossible=False)
    data = result.data

    # --- statistics on the cleaned rows (complete cases, nothing filled) ---
    log = stats.TestLog()
    table = stats.section_table(data)
    pair = stats.welch_pair(data, *PRE_CHOSEN_PAIR, log)
    anova = stats.anova_sections(data, log)
    corr = stats.correlations(data, "hours", "final", log, seed=seed)
    corr_naive = stats.correlations(naive.data, "hours", "final", None, seed=seed)
    hours_prior = stats.correlations(data, "hours", "prior", None, seed=seed, n_boot=200)
    adjusted = stats.adjusted_hours_slope(data, seed=seed)
    tests = log.holm()

    # --- split FIRST, then fill and standardise from train only ---
    train, test = model.train_test_split(data, 0.2, seed)
    filler = model.MedianFiller().fit(train)
    train_f, test_f = filler.transform(train), filler.transform(test)
    all_medians = {c: float(data[c].median()) for c in filler.columns}
    scaler = model.Standardiser().fit(train_f["prior"].to_numpy())
    z_train, z_test = scaler.transform(train_f["prior"]), scaler.transform(test_f["prior"])
    y_train, y_test = train_f["final"].to_numpy(float), test_f["final"].to_numpy(float)

    curves = {}
    for rate in GD_RATES:
        curves[rate] = model.gradient_descent(z_train, y_train, rate, GD_STEPS)[2]
    w0, w1, _ = model.gradient_descent(z_train, y_train, MAIN_RATE, GD_STEPS)
    l0, l1 = model.least_squares(z_train, y_train)
    pred = w0 + w1 * z_test
    baseline = np.full(len(y_test), y_train.mean())
    fit = {
        "gd_intercept": w0, "gd_slope": w1, "lstsq_intercept": l0, "lstsq_slope": l1,
        "max_difference": max(abs(w0 - l0), abs(w1 - l1)), "rate": MAIN_RATE, "steps": GD_STEPS,
        "final_cost": curves[MAIN_RATE][-1], "first_cost": curves[MAIN_RATE][0],
        "test_rmse": model.rmse(y_test, pred), "baseline_rmse": model.rmse(y_test, baseline),
        "n_train": len(train), "n_test": len(test), "prior_mean_train": scaler.mean_, "prior_sd_train": scaler.sd_,
        "train_medians": filler.medians_, "all_row_medians": all_medians,
        "gaps_filled_train": {c: int(train_f[c + "_was_gap"].sum()) for c in filler.columns},
        "gaps_filled_test": {c: int(test_f[c + "_was_gap"].sum()) for c in filler.columns},
    }

    # --- write outputs (never into data/) ---
    table.to_csv(out_dir / "section_table.csv", index=False)
    result.removed.to_csv(out_dir / "removed_rows.csv", index=False)
    pd.DataFrame(result.log).to_csv(out_dir / "cleaning_log.csv", index=False)
    data.to_csv(out_dir / "clean_students.csv", index=False)
    train_f.to_csv(out_dir / "train_filled.csv", index=False)
    test_f.to_csv(out_dir / "test_filled.csv", index=False)
    tests.to_csv(out_dir / "tests_run.csv", index=False)
    if make_charts:
        charts.scatter_hours_final(data, corr, out_dir / "charts" / "hours_vs_final.png")
        charts.section_means(table, out_dir / "charts" / "section_means.png")
        charts.cost_curves(curves, out_dir / "charts" / "gradient_descent_cost.png")

    summary = {"raw_rows": result.n_raw, "analysis_rows": len(data), "removed_rows": len(result.removed), "section_table": table.to_dict("records"),
               "pair": pair, "anova": anova, "corr": corr, "corr_naive": corr_naive, "hours_prior": hours_prior, "adjusted": adjusted,
               "tests_run": log.n, "fit": fit}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out_dir / "report.md").write_text(build_report(result, naive, table, pair, anova, corr, corr_naive, hours_prior, adjusted, tests, fit, sha_before), encoding="utf-8")

    assert_unchanged(raw_path, sha_before)   # the raw file must be byte-for-byte the same as when we started
    summary["raw_sha256"] = sha_before
    return summary
