"""Break the code on purpose, one change at a time, and check the tests notice.   python break_it.py   (takes about 5 minutes)
Every change must make at least one test fail. A line that says NOT CAUGHT means a test is missing."""
import pathlib, shutil, subprocess, sys, tempfile
SRC = pathlib.Path(__file__).resolve().parent
MUT = [
 ("study/clean.py", 'keep="first")\n    first', 'keep=False)\n    first', "repeats: drop none of them (keep=False)"),
 ("study/clean.py", "if blank_impossible:\n                df.loc[bad, col] = np.nan", "if False:\n                df.loc[bad, col] = np.nan", "impossible values never blanked"),
 ("study/clean.py", '"hours": (0.0, 100.0)', '"hours": (0.0, 1000.0)', "hours limit raised to 1000 (940 survives)"),
 ("study/clean.py", "df = df[~nofinal].copy()", "df = df.copy()", "rows with no final are kept"),
 ("study/clean.py", '(hrs?|h|hours?|%|pts?|points?)$")', '(hrs?|h|hours?|pts?|points?)$")', "'78%' no longer readable"),
 ("study/clean.py", 're.sub(r"(section|sec)", "", str(text).strip().lower())', 'str(text).strip().lower()', "'Sec b' not understood"),
 ("study/clean.py", 'note("repeat rows removed (exact copy)", len(exact)', 'note("repeat rows removed (exact copy)", 0', "repeat count lies (0)"),
 ("study/load.py", "if now != expected_sha:", "if False:", "the unchanged-check never fires"),
 ("study/load.py", "dtype=str, keep_default_na=False", "dtype=str", "NA turned into NaN by pandas at load"),
 ("study/stats.py", "sd = x.std(ddof=1) if n > 1 else np.nan", "sd = x.std(ddof=0) if n > 1 else np.nan", "sd uses n instead of n-1"),
 ("study/stats.py", "stats.t.interval(confidence, df=n - 1, loc=mean", "stats.norm.interval(confidence, loc=mean", "normal interval instead of t"),
 ("study/stats.py", "(m - rank) * df", "m * df", "Holm turned into Bonferroni"),
 ("study/stats.py", "running = max(running, min(1.0, (m - rank) * df[\"p\"].iloc[i]))", "running = min(1.0, (m - rank) * df[\"p\"].iloc[i])", "Holm: no running maximum"),
 ("study/stats.py", "equal_var=False)\n    log.add(f\"Welch", "equal_var=True)\n    log.add(f\"Welch", "pair test assumes equal variances"),
 ("study/stats.py", "sr = stats.spearmanr(xs, ys)", "sr = stats.pearsonr(xs, ys)", "Spearman computed as Pearson"),
 ("study/stats.py", "boots.append(np.linalg.lstsq(X[s], y[s], rcond=None)[0][1])", "boots.append(np.linalg.lstsq(X, y, rcond=None)[0][1])", "bootstrap does not resample"),
 ("study/model.py", "train_idx = order[:n_test], order[n_test:]", "train_idx = order[:n_test], order[n_test - 1:]", "x"),
 ("study/model.py", "test_idx, train_idx = order[:n_test], order[n_test:]", "test_idx, train_idx = order[:n_test], order[n_test - 1:]", "split: one row in both train and test"),
 ("study/model.py", "self.sd_ = float(np.mean(x)), float(np.std(x, ddof=0))", "self.sd_ = float(np.mean(x)), float(np.std(x, ddof=1))", "x"),
 ("study/model.py", "float(np.std(x, ddof=0))", "float(np.std(x, ddof=1))", "standardise with n-1"),
 ("study/model.py", "w1 -= rate * np.sum(err * z) / n", "w1 -= rate * np.sum(err) / n", "gradient for the slope ignores z"),
 ("study/model.py", "w0 -= rate * np.sum(err) / n", "w0 -= rate * np.sum(err)", "gradient for the intercept forgets /n"),
 ("study/model.py", "for _ in range(steps + 1):", "for _ in range(steps):", "one cost point missing"),
 ("study/model.py", "out[c] = out[c].fillna(m)", "out[c] = out[c].fillna(0)", "gaps filled with 0"),
 ("study/pipeline.py", "filler = model.MedianFiller().fit(train)", "filler = model.MedianFiller().fit(data)", "fill medians from ALL rows (test leaks in)"),
 ("study/pipeline.py", "scaler = model.Standardiser().fit(train_f[\"prior\"].to_numpy())", "scaler = model.Standardiser().fit(pd.concat([train_f, test_f])[\"prior\"].to_numpy())", "standardise with test rows too"),
 ("study/pipeline.py", "    assert_unchanged(raw_path, sha_before)", "    pass", "pipeline never re-checks the raw file"),
 ("study/pipeline.py", "pair = stats.welch_pair(data, *PRE_CHOSEN_PAIR, log)", "pair = stats.welch_pair(data, *PRE_CHOSEN_PAIR, stats.TestLog())", "pair test hidden from the test count"),
 ("study/report.py", 'if p_pearson < 0.05 and corr["pearson_low"] > 0:', 'if p_pearson < 0.05 or corr["pearson_low"] > 0:', "answer: 'or' instead of 'and'"),
 ("study/report.py", 'p_pearson = holm["Pearson correlation, hours vs final"]', 'p_pearson = tests.set_index("test").loc["Pearson correlation, hours vs final", "p"]', "answer uses the uncorrected p"),
 ("study/report.py", 'A(f"**{len(tests)} tests were run**', 'A(f"**3 tests were run**', "report says the wrong number of tests"),
]
EQUIVALENT = {"NA turned into NaN by pandas at load"}
def main():
    bad = 0
    for path, old, new, label in MUT:
        if label == "x": continue
        tmp = pathlib.Path(tempfile.mkdtemp()) / "p"
        shutil.copytree(SRC, tmp, ignore=shutil.ignore_patterns("outputs", "__pycache__", ".venv"))
        f = tmp / path
        text = f.read_text()
        if old not in text:
            print(f"SKIPPED (text not found)  {label}"); bad += 1; continue
        f.write_text(text.replace(old, new, 1))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider"], cwd=tmp, capture_output=True, text=True)
        caught = r.returncode != 0
        tag = "caught" if caught else ("NOT CAUGHT" )
        first = next((l for l in r.stdout.splitlines() if l.startswith("FAILED")), "")[:110]
        print(f"{tag:11} {label:55} {first}")
        if not caught: bad += 1
        shutil.rmtree(tmp.parent)
    print(f"\n{len([m for m in MUT if m[3]!='x'])-bad} caught, {bad} not caught")
main()
