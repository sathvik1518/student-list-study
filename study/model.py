"""Train/test split, gap filling from TRAIN only, and gradient descent written by hand."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def train_test_split(data: pd.DataFrame, test_fraction: float = 0.2, seed: int = 51) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle once with a fixed seed, then cut. The test rows are put away BEFORE any median, mean or fill is computed."""
    order = np.random.default_rng(seed).permutation(len(data))
    n_test = int(round(len(data) * test_fraction))
    test_idx, train_idx = order[:n_test], order[n_test:]
    return data.iloc[np.sort(train_idx)].reset_index(drop=True), data.iloc[np.sort(test_idx)].reset_index(drop=True)


@dataclass
class MedianFiller:
    """Fit on train. Apply to train AND test with the TRAIN medians. Test rows never influence a fill value."""
    columns: tuple[str, ...] = ("hours", "prior")

    def fit(self, train: pd.DataFrame) -> "MedianFiller":
        self.medians_ = {c: float(train[c].median()) for c in self.columns}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c, m in self.medians_.items():
            out[c + "_was_gap"] = out[c].isna()
            out[c] = out[c].fillna(m)
        return out


@dataclass
class Standardiser:
    mean_: float = 0.0
    sd_: float = 1.0

    def fit(self, x: np.ndarray) -> "Standardiser":
        self.mean_, self.sd_ = float(np.mean(x)), float(np.std(x, ddof=0))
        if self.sd_ == 0:
            raise ValueError("cannot standardise a column that never changes")
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, float) - self.mean_) / self.sd_


def gradient_descent(z: np.ndarray, y: np.ndarray, rate: float, steps: int) -> tuple[float, float, list[float]]:
    """Fit y = w0 + w1*z by walking downhill on cost = (1/2n) * sum((w0 + w1*z - y)^2). Returns w0, w1 and the cost at every step (step 0 = before moving)."""
    z, y = np.asarray(z, float), np.asarray(y, float)
    n = len(y)
    w0 = w1 = 0.0
    costs = []
    for _ in range(steps + 1):
        err = w0 + w1 * z - y
        costs.append(float(np.sum(err ** 2) / (2 * n)))
        w0 -= rate * np.sum(err) / n
        w1 -= rate * np.sum(err * z) / n
    # costs has steps+1 entries (before each move, plus the last point); w0/w1 are the values after `steps` moves
    return float(w0), float(w1), costs


def least_squares(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    X = np.column_stack([np.ones(len(z)), z])
    w = np.linalg.lstsq(X, np.asarray(y, float), rcond=None)[0]
    return float(w[0]), float(w[1])


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true, float) - np.asarray(y_pred, float)) ** 2)))
