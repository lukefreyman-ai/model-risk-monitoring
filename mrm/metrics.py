"""Monitoring metrics: PSI, CSI, KS, Gini/AUC, calibration (incl. Hosmer-Lemeshow), score-band drift."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve


def psi(expected: np.ndarray, actual: np.ndarray, bins: int | np.ndarray = 10, eps: float = 1e-4) -> tuple[float, pd.DataFrame]:
    """Population Stability Index between two samples of a continuous variable (bins from the expected sample's deciles).
    Rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1))) if isinstance(bins, int) else np.asarray(bins)
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected); a = np.histogram(actual, edges)[0] / len(actual)
    e = np.clip(e, eps, None); a = np.clip(a, eps, None)
    contrib = (a - e) * np.log(a / e)
    table = pd.DataFrame({"bin_low": edges[:-1], "bin_high": edges[1:], "expected_pct": e, "actual_pct": a, "psi": contrib})
    return float(contrib.sum()), table


def csi(expected_bins: pd.Series, actual_bins: pd.Series, eps: float = 1e-4) -> tuple[float, pd.DataFrame]:
    """Characteristic Stability Index: PSI applied to a feature's scorecard bins (categorical)."""
    cats = expected_bins.cat.categories if hasattr(expected_bins, "cat") else sorted(set(expected_bins) | set(actual_bins), key=str)
    e = expected_bins.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy(); a = actual_bins.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    e = np.clip(e, eps, None); a = np.clip(a, eps, None)
    contrib = (a - e) * np.log(a / e)
    return float(contrib.sum()), pd.DataFrame({"bin": [str(c) for c in cats], "expected_pct": e, "actual_pct": a, "csi": contrib})


def ks_stat(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def gini(y: np.ndarray, p: np.ndarray) -> float:
    return float(2 * roc_auc_score(y, p) - 1)


def calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile calibration: predicted vs. actual default rate per score decile."""
    q = pd.qcut(p, n_bins, labels=False, duplicates="drop")
    t = pd.DataFrame({"decile": q, "y": y, "p": p}).groupby("decile").agg(n=("y", "size"), actual=("y", "mean"), predicted=("p", "mean"))
    t["gap"] = t["actual"] - t["predicted"]
    return t.reset_index()


def hosmer_lemeshow(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """H-L chi-square statistic and p-value (p < 0.05 -> evidence of miscalibration). Large samples make this strict."""
    t = calibration_table(y, p, n_bins)
    obs1 = t["actual"] * t["n"]; exp1 = t["predicted"] * t["n"]; obs0 = t["n"] - obs1; exp0 = t["n"] - exp1
    chi2 = float((((obs1 - exp1) ** 2) / exp1.clip(lower=1e-9) + ((obs0 - exp0) ** 2) / exp0.clip(lower=1e-9)).sum())
    dof = max(len(t) - 2, 1)
    return chi2, float(1 - stats.chi2.cdf(chi2, dof))


def score_bands(scores: np.ndarray, y: np.ndarray, edges: list[float]) -> pd.DataFrame:
    b = pd.cut(scores, edges)
    t = pd.DataFrame({"band": b, "y": y}).groupby("band", observed=False)["y"].agg(["size", "mean"]).rename(columns={"size": "n", "mean": "bad_rate"})
    t["share"] = t["n"] / t["n"].sum()
    return t.reset_index()


def rag(value: float, green: float, amber: float, higher_is_worse: bool = True) -> str:
    if higher_is_worse:
        return "GREEN" if value < green else "AMBER" if value < amber else "RED"
    return "GREEN" if value > green else "AMBER" if value > amber else "RED"
