"""The statistics. Every test goes through a TestLog so we can say how many tests we ran BEFORE trusting a p-value."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestLog:
    rows: list[dict] = field(default_factory=list)

    def add(self, name: str, statistic: float, p: float) -> None:
        self.rows.append({"test": name, "statistic": float(statistic), "p": float(p)})

    @property
    def n(self) -> int:
        return len(self.rows)

    def holm(self) -> pd.DataFrame:
        """Holm's correction: with m tests, the smallest p is multiplied by m, the next by m-1, and so on (never above 1, never going down)."""
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df.assign(p_holm=[])
        order = np.argsort(df["p"].to_numpy())
        m = len(df)
        adjusted = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, min(1.0, (m - rank) * df["p"].iloc[i]))
            adjusted[i] = running
        df["p_holm"] = adjusted
        return df


def section_table(data: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    rows = []
    for sec, g in data[data["section"] != "Unknown"].groupby("section"):
        x = g["final"].to_numpy(float)
        n = len(x)
        mean = x.mean()
        sd = x.std(ddof=1) if n > 1 else np.nan
        if n > 1:
            lo, hi = stats.t.interval(confidence, df=n - 1, loc=mean, scale=sd / np.sqrt(n))
        else:
            lo = hi = np.nan
        rows.append({"section": sec, "n": n, "mean_final": mean, "sd": sd, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def welch_pair(data: pd.DataFrame, a: str, b: str, log: TestLog, confidence: float = 0.95) -> dict:
    """ONE pre-chosen pair (decided before looking at the numbers). Difference = mean(a) - mean(b), with a Welch confidence interval."""
    xa = data.loc[data["section"] == a, "final"].to_numpy(float)
    xb = data.loc[data["section"] == b, "final"].to_numpy(float)
    va, vb = xa.var(ddof=1) / len(xa), xb.var(ddof=1) / len(xb)
    diff = xa.mean() - xb.mean()
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(xa) - 1) + vb ** 2 / (len(xb) - 1))
    half = stats.t.ppf((1 + confidence) / 2, dof) * se
    t = stats.ttest_ind(xa, xb, equal_var=False)
    log.add(f"Welch t-test, section {a} vs {b}", t.statistic, t.pvalue)
    return {"a": a, "b": b, "diff": diff, "ci_low": diff - half, "ci_high": diff + half, "t": float(t.statistic), "p": float(t.pvalue), "dof": float(dof)}


def anova_sections(data: pd.DataFrame, log: TestLog) -> dict:
    groups = [g["final"].to_numpy(float) for _, g in data[data["section"] != "Unknown"].groupby("section")]
    res = stats.f_oneway(*groups)
    log.add("One-way ANOVA, final by section", res.statistic, res.pvalue)
    return {"F": float(res.statistic), "p": float(res.pvalue)}


def complete_pairs(data: pd.DataFrame, a: str, b: str) -> tuple[np.ndarray, np.ndarray]:
    sub = data[[a, b]].dropna()
    return sub[a].to_numpy(float), sub[b].to_numpy(float)


def correlations(data: pd.DataFrame, x: str, y: str, log: TestLog | None = None, seed: int = 51, n_boot: int = 2000) -> dict:
    """Pearson (straight-line link, shaken by outliers) and Spearman (ranks, steady against outliers), each with a 95% interval."""
    xs, ys = complete_pairs(data, x, y)
    pr = stats.pearsonr(xs, ys)
    pci = pr.confidence_interval(0.95)
    sr = stats.spearmanr(xs, ys)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(xs))
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots.append(stats.spearmanr(xs[s], ys[s]).statistic)
    slo, shi = np.percentile(boots, [2.5, 97.5])
    if log is not None:
        log.add(f"Pearson correlation, {x} vs {y}", pr.statistic, pr.pvalue)
        log.add(f"Spearman correlation, {x} vs {y}", sr.statistic, sr.pvalue)
    return {"n": len(xs), "pearson": float(pr.statistic), "pearson_low": float(pci.low), "pearson_high": float(pci.high), "pearson_p": float(pr.pvalue),
            "spearman": float(sr.statistic), "spearman_low": float(slo), "spearman_high": float(shi), "spearman_p": float(sr.pvalue)}


def adjusted_hours_slope(data: pd.DataFrame, seed: int = 51, n_boot: int = 2000) -> dict:
    """Slope of final on hours AFTER taking prior into account (least squares), with a bootstrap 95% interval. Not a hypothesis test, so it is not counted in the TestLog."""
    sub = data[["hours", "prior", "final"]].dropna()
    X = np.column_stack([np.ones(len(sub)), sub["hours"], sub["prior"]])
    y = sub["final"].to_numpy(float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.integers(0, len(sub), len(sub))
        boots.append(np.linalg.lstsq(X[s], y[s], rcond=None)[0][1])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(sub), "slope": float(beta[1]), "low": float(lo), "high": float(hi), "prior_slope": float(beta[2])}
