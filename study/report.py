"""Builds outputs/report.md. The ANSWER wording follows the numbers (it is chosen by rules, never typed in advance)."""
import pandas as pd


def _md(df: pd.DataFrame, fmt: dict | None = None) -> str:
    fmt = fmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(fmt[c](v) if c in fmt and pd.notna(v) else ("" if pd.isna(v) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _p(v: float) -> str:
    return "<0.001" if v < 0.001 else f"{v:.3f}"


def answer_lines(table, pair, anova, corr, adjusted, hours_prior, tests) -> list[str]:
    holm = {row["test"]: row["p_holm"] for _, row in tests.iterrows()}
    p_anova = holm["One-way ANOVA, final by section"]
    p_pearson = holm["Pearson correlation, hours vs final"]
    out = []
    # hours
    if p_pearson < 0.05 and corr["pearson_low"] > 0:
        out.append(f"**Hours: yes, they go together.** Students who study more hours tend to have a higher final score (Pearson {corr['pearson']:.2f}, 95% interval {corr['pearson_low']:.2f} to {corr['pearson_high']:.2f}; "
                   f"Spearman {corr['spearman']:.2f}). With the starting level (prior score) taken into account, one more weekly hour goes with about {adjusted['slope']:.2f} more points "
                   f"(95% interval {adjusted['low']:.2f} to {adjusted['high']:.2f}).")
    else:
        out.append(f"**Hours: no clear link in this data.** Pearson {corr['pearson']:.2f} (95% interval {corr['pearson_low']:.2f} to {corr['pearson_high']:.2f}). The interval includes 0 or the corrected p-value is not below 0.05.")
    out.append("**But this is not proof that extra hours CAUSE higher scores.** Nobody was told to study more. Students who started stronger may study more "
               f"(hours vs prior: Pearson {hours_prior['pearson']:.2f}). The prior-adjusted number above reduces that problem, it does not remove it.")
    # section
    if p_anova < 0.05:
        out.append(f"**Section: it seems to matter.** The averages differ more than chance alone usually explains (ANOVA p = {_p(anova['p'])}, corrected {_p(p_anova)}).")
    else:
        out.append(f"**Section: no clear sign that it matters (yet).** The averages differ a little (ANOVA p = {_p(anova['p'])}), but after correcting for the {len(tests)} tests that were run the evidence is not strong enough (corrected p = {_p(p_anova)}). "
                   f"For the one pair chosen in advance, A minus B is {pair['diff']:.1f} points (95% interval {pair['ci_low']:.1f} to {pair['ci_high']:.1f}). "
                   "An interval that includes 0 means 'we cannot tell', not 'no difference'.")
    return out


def build_report(result, naive, table, pair, anova, corr, corr_naive, hours_prior, adjusted, tests, fit, sha) -> str:
    removed = result.removed
    reasons = removed["reason"].value_counts().rename_axis("reason").reset_index(name="rows")
    n_gap = {c: int(result.data[c].isna().sum()) for c in ("hours", "prior")}
    unknown = int((result.data["section"] == "Unknown").sum())
    lines = []
    A = lines.append

    A("# Do extra study hours help, and does the section matter?")
    A("")
    A("## The answer (one page)")
    A("")
    for s in answer_lines(table, pair, anova, corr, adjusted, hours_prior, tests):
        A(f"- {s}")
    A("")
    A(f"Based on {len(result.data)} students out of {result.n_raw} rows in the raw file. Everything below shows how that number was reached and what could not be checked.")
    A("")
    A("### Average final score by section (95% interval)")
    A("")
    A(_md(table, {"mean_final": lambda v: f"{v:.1f}", "sd": lambda v: f"{v:.1f}", "ci_low": lambda v: f"{v:.1f}", "ci_high": lambda v: f"{v:.1f}"}))
    A("")
    A("![Average final score by section](charts/section_means.png)")
    A("")
    A("### Hours against final score")
    A("")
    A(f"- Pearson (straight-line link) {corr['pearson']:.2f}, 95% interval {corr['pearson_low']:.2f} to {corr['pearson_high']:.2f}")
    A(f"- Spearman (ranks, not bothered by odd values) {corr['spearman']:.2f}, 95% interval {corr['spearman_low']:.2f} to {corr['spearman_high']:.2f} (bootstrap)")
    A(f"- Students with both numbers: {corr['n']}")
    A("")
    A("![Hours vs final](charts/hours_vs_final.png)")
    A("")
    A("## What was found and fixed (every number is a count)")
    A("")
    A(f"Raw file: **{result.n_raw}** rows. It was read as text and never changed (sha256 `{sha[:16]}...` checked before and after).")
    A("")
    A(_md(pd.DataFrame(result.log)[["action", "count", "why", "examples"]]))
    A("")
    A("### Rows that were removed (all of them, with the reason)")
    A("")
    A(_md(reasons))
    A("")
    A(f"Check: {result.n_raw} raw rows = {len(removed)} removed + {len(result.data)} used. The list is in `outputs/removed_rows.csv`.")
    A("")
    A("### The 940-hour row: with and without")
    A("")
    cmp = pd.DataFrame([
        {"version": "naive: impossible values KEPT", "n": corr_naive["n"], "pearson": f"{corr_naive['pearson']:.2f}", "spearman": f"{corr_naive['spearman']:.2f}"},
        {"version": "cleaned: impossible values blanked", "n": corr["n"], "pearson": f"{corr['pearson']:.2f}", "spearman": f"{corr['spearman']:.2f}"},
    ])
    A(_md(cmp))
    A("")
    dp, ds = abs(corr["pearson"] - corr_naive["pearson"]), abs(corr["spearman"] - corr_naive["spearman"])
    A(f"Keeping the impossible values moved Pearson by {dp:.2f} and Spearman by {ds:.2f}. " + ("One huge value pulls a straight line a lot; ranks barely notice it. That is why both are reported." if dp > ds else "Here neither moved much."))
    A("")
    A("## Tests run: how many, before trusting a p-value")
    A("")
    A(f"**{len(tests)} tests were run**, and they are all listed. Because several were run, each p-value is also corrected with Holm's method (`p_holm`). Trust `p_holm`, not `p`.")
    A("")
    A(_md(tests, {"statistic": lambda v: f"{v:.3f}", "p": _p, "p_holm": _p}))
    A("")
    A("The pair A vs B was chosen before looking at any section averages. Looking first and then testing the biggest gap would be a hidden extra test.")
    A("")
    A("## Train / test split and gradient descent")
    A("")
    A(f"- The test rows ({fit['n_test']} students, 20%) were put away first. Train: {fit['n_train']} students.")
    A(f"- Gaps in `hours` / `prior` were filled with the **train** median only: hours {fit['train_medians']['hours']:.2f}, prior {fit['train_medians']['prior']:.1f}. "
       f"(Median of ALL rows would have been hours {fit['all_row_medians']['hours']:.2f}, prior {fit['all_row_medians']['prior']:.1f}. The test rows are not allowed to influence a fill value.)")
    A(f"- Gaps filled: train {fit['gaps_filled_train']}, test {fit['gaps_filled_test']}.")
    A(f"- `prior` was standardised with the train mean {fit['prior_mean_train']:.2f} and train sd {fit['prior_sd_train']:.2f}.")
    A(f"- Gradient descent for `final` on standardised `prior`, written from scratch: rate {fit['rate']}, {fit['steps']} steps. Cost went from {fit['first_cost']:.1f} to {fit['final_cost']:.2f}.")
    A(f"- Check against `numpy.linalg.lstsq` on the same train rows: intercept {fit['gd_intercept']:.4f} vs {fit['lstsq_intercept']:.4f}, slope {fit['gd_slope']:.4f} vs {fit['lstsq_slope']:.4f}. Biggest difference {fit['max_difference']:.1e}.")
    A(f"- On the {fit['n_test']} test students: RMSE {fit['test_rmse']:.2f}, against {fit['baseline_rmse']:.2f} for always guessing the train average.")
    A("")
    A("![Gradient descent cost](charts/gradient_descent_cost.png)")
    A("")
    A("## What I did not check")
    A("")
    items = [
        "**Cause.** This is an observed list, not an experiment. I cannot say extra hours cause higher scores.",
        "**Who wrote the hours.** I do not know if students reported them honestly or how they were counted.",
        f"**The true values of the impossible numbers.** They were turned into gaps, not corrected. I do not know what they were meant to be.",
        f"**The repeated student with different numbers.** I kept the first row. I cannot know which one is right.",
        f"**Gaps may not be random.** If students who study little skip the `hours` question, the filled median hides that. Gaps left in `hours`: {n_gap['hours']}, in `prior`: {n_gap['prior']} (these rows are skipped in the correlations, filled only in the train/test model).",
        *( [f"**{unknown} students with an unknown section** are left out of the section table."] if unknown else [] ),
        "**A curve.** I only looked at straight-line links. A curve (for example, hours helping up to a point) would not show up here.",
        "**Normal-shaped scores.** The intervals assume scores are roughly bell-shaped. I did not test that.",
        f"**A small test set.** {fit['n_test']} test students make the test RMSE noisy.",
        "**More than one pre-chosen pair.** Only A vs B was tested. B vs C and A vs C were not.",
        "**Where the file came from.** `data/students_raw.csv` is made by `make_raw_data.py` as a stand-in. If your lesson has its own `students_raw.csv`, results will differ and the column names must match.",
    ]
    for it in items:
        A(f"- {it}")
    A("")
    return "\n".join(lines)
