import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from study import stats


def make(sections, finals, hours=None, prior=None):
    n = len(finals)
    return pd.DataFrame({"student_id": [f"S{i}" for i in range(n)], "section": sections, "final": finals,
                         "hours": hours if hours is not None else np.arange(n, dtype=float), "prior": prior if prior is not None else np.arange(n, dtype=float)})


def test_section_table_matches_a_hand_calculation():
    d = make(["A"] * 4 + ["B"] * 3, [60, 70, 80, 90, 50, 55, 60])
    t = stats.section_table(d).set_index("section")
    assert t.loc["A", "n"] == 4 and t.loc["A", "mean_final"] == 75
    sd = np.std([60, 70, 80, 90], ddof=1)
    half = sps.t.ppf(0.975, 3) * sd / 2
    assert t.loc["A", "ci_low"] == pytest.approx(75 - half) and t.loc["A", "ci_high"] == pytest.approx(75 + half)
    assert t.loc["B", "mean_final"] == pytest.approx(55)


def test_section_table_leaves_unknown_out_and_survives_a_group_of_one():
    d = make(["A", "A", "Unknown", "B"], [60, 70, 99, 50])
    t = stats.section_table(d).set_index("section")
    assert list(t.index) == ["A", "B"]
    assert np.isnan(t.loc["B", "ci_low"]) and t.loc["B", "n"] == 1


def test_a_95_percent_interval_covers_the_true_mean_about_95_percent_of_the_time():
    rng = np.random.default_rng(0)
    hits = 0
    runs = 2000
    for _ in range(runs):
        d = make(["A"] * 30, rng.normal(50, 10, 30))
        row = stats.section_table(d).iloc[0]
        hits += row["ci_low"] <= 50 <= row["ci_high"]
    assert 0.93 <= hits / runs <= 0.97


def test_welch_pair_matches_scipy_and_logs_one_test():
    rng = np.random.default_rng(1)
    a, b = rng.normal(70, 10, 40), rng.normal(65, 14, 55)
    d = make(["A"] * 40 + ["B"] * 55, np.concatenate([a, b]))
    log = stats.TestLog()
    res = stats.welch_pair(d, "A", "B", log)
    ref = sps.ttest_ind(a, b, equal_var=False)
    assert res["t"] == pytest.approx(ref.statistic) and res["p"] == pytest.approx(ref.pvalue)
    assert res["diff"] == pytest.approx(a.mean() - b.mean())
    assert res["ci_low"] < res["diff"] < res["ci_high"]
    ref_ci = ref.confidence_interval(0.95)
    assert res["ci_low"] == pytest.approx(ref_ci.low) and res["ci_high"] == pytest.approx(ref_ci.high)
    assert log.n == 1


def test_anova_matches_scipy():
    rng = np.random.default_rng(2)
    g = [rng.normal(m, 8, 30) for m in (60, 62, 70)]
    d = make(["A"] * 30 + ["B"] * 30 + ["C"] * 30, np.concatenate(g))
    log = stats.TestLog()
    res = stats.anova_sections(d, log)
    ref = sps.f_oneway(*g)
    assert res["F"] == pytest.approx(ref.statistic) and res["p"] == pytest.approx(ref.pvalue) and log.n == 1


def test_holm_known_example():
    log = stats.TestLog()
    for name, p in [("a", 0.01), ("b", 0.04), ("c", 0.03)]:
        log.add(name, 0.0, p)
    out = log.holm().set_index("test")["p_holm"]
    assert out["a"] == pytest.approx(0.03) and out["c"] == pytest.approx(0.06) and out["b"] == pytest.approx(0.06)


def test_holm_never_lowers_a_p_value_never_passes_one_and_keeps_the_order():
    rng = np.random.default_rng(3)
    for _ in range(50):
        log = stats.TestLog()
        for i, p in enumerate(rng.uniform(0, 1, rng.integers(1, 9))):
            log.add(str(i), 0.0, p)
        df = log.holm()
        assert (df["p_holm"] >= df["p"] - 1e-12).all() and (df["p_holm"] <= 1).all()
        ordered = df.sort_values("p")["p_holm"].to_numpy()
        assert (np.diff(ordered) >= -1e-12).all()


def test_holm_with_one_test_changes_nothing_and_with_none_is_empty():
    log = stats.TestLog()
    assert len(log.holm()) == 0
    log.add("x", 1.0, 0.2)
    assert log.holm().loc[0, "p_holm"] == pytest.approx(0.2)


def test_correlations_match_scipy_and_skip_gaps():
    rng = np.random.default_rng(4)
    x = rng.normal(8, 3, 120)
    y = 40 + 2 * x + rng.normal(0, 6, 120)
    d = make(["A"] * 120, y, hours=x)
    d.loc[[3, 10], "hours"] = np.nan
    log = stats.TestLog()
    c = stats.correlations(d, "hours", "final", log, n_boot=300)
    keep = d["hours"].notna()
    assert c["n"] == 118
    assert c["pearson"] == pytest.approx(sps.pearsonr(x[keep], y[keep]).statistic)
    assert c["spearman"] == pytest.approx(sps.spearmanr(x[keep], y[keep]).statistic)
    assert c["pearson_low"] < c["pearson"] < c["pearson_high"]
    assert c["spearman_low"] < c["spearman"] < c["spearman_high"]
    assert log.n == 2


def test_one_huge_value_wrecks_pearson_much_more_than_spearman():
    rng = np.random.default_rng(5)
    x = rng.normal(8, 3, 100)
    y = 40 + 2 * x + rng.normal(0, 5, 100)
    clean_c = stats.correlations(make(["A"] * 100, y, hours=x), "hours", "final", n_boot=50)
    x2 = x.copy(); x2[0] = 940
    dirty_c = stats.correlations(make(["A"] * 100, y, hours=x2), "hours", "final", n_boot=50)
    assert abs(dirty_c["pearson"] - clean_c["pearson"]) > 0.3
    assert abs(dirty_c["spearman"] - clean_c["spearman"]) < 0.05


def test_correlation_is_repeatable_with_the_same_seed():
    rng = np.random.default_rng(6)
    d = make(["A"] * 60, rng.normal(60, 10, 60), hours=rng.normal(8, 3, 60))
    assert stats.correlations(d, "hours", "final", n_boot=100, seed=1) == stats.correlations(d, "hours", "final", n_boot=100, seed=1)


def test_adjusted_slope_recovers_a_known_slope():
    rng = np.random.default_rng(7)
    prior = rng.normal(60, 12, 400)
    hours = 8 + 0.1 * (prior - 60) + rng.normal(0, 3, 400)
    final = 10 + 0.4 * prior + 2.0 * hours + rng.normal(0, 3, 400)
    res = stats.adjusted_hours_slope(make(["A"] * 400, final, hours=hours, prior=prior), n_boot=300)
    assert res["low"] < 2.0 < res["high"] and res["slope"] == pytest.approx(2.0, abs=0.2) and res["prior_slope"] == pytest.approx(0.4, abs=0.05)
