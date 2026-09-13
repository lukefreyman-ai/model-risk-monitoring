"""Give Me Some Credit loader + simulated monitoring windows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KAGGLE_URL = "https://www.kaggle.com/c/GiveMeSomeCredit/data"


def load_gmsc(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Download cs-training.csv from {KAGGLE_URL} (Kaggle login) and place it there.")
    df = pd.read_csv(p)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def assign_windows(df: pd.DataFrame, cfg: dict, seed: int = 42) -> pd.Series:
    """Assign each row to a window: 0 = development (unbiased random 40 %), 1..n-1 = monitoring periods, -1 = unused.

    Monitoring rows are first spread uniformly over windows 1..n-1, then each window is *thinned*: a row is kept with
    probability proportional to
        (1 + util)^(utilization_tilt[w] - 1) * (age / 45)^(age_tilt[w] - 1) * bad_rate_tilt[w]^y
    so later windows over-represent high-utilisation, younger and (via bad_rate_tilt) worse-performing applicants,
    each window independently of the others. All tilts live in config.yaml and are quoted in the validation report."""
    rng = np.random.default_rng(seed)
    n_w = int(cfg["n_windows"]); n = len(df)
    util = df["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1.5).fillna(0).to_numpy()
    age = df["age"].clip(lower=18).fillna(45).to_numpy()
    target = cfg.get("target", "SeriousDlqin2yrs")
    y = df[target].to_numpy() if target in df else np.zeros(n)
    tilt_u = np.array(cfg["drift"]["utilization_tilt"], dtype=float); tilt_a = np.array(cfg["drift"]["age_tilt"], dtype=float)
    tilt_b = np.array(cfg["drift"].get("bad_rate_tilt", [1.0] * n_w), dtype=float)
    win = np.where(rng.random(n) < 0.40, 0, rng.integers(1, n_w, n))
    keep = np.ones(n, dtype=bool)
    for w in range(1, n_w):
        m = win == w
        weight = (1 + util[m]) ** (tilt_u[w] - 1) * (age[m] / 45.0) ** (tilt_a[w] - 1) * tilt_b[w] ** y[m]
        keep[m] = rng.random(m.sum()) < np.minimum(1.0, weight / np.quantile(weight, 0.90))   # keep ~all of the top decile, thin the rest
    return pd.Series(np.where(keep, win, -1), index=df.index, name="window")


def make_synthetic(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the GMSC schema for tests (never for reported numbers)."""
    rng = np.random.default_rng(seed)
    util = np.clip(rng.beta(1.2, 3, n) * 1.2, 0, 1.5); age = rng.integers(21, 85, n)
    late30 = rng.poisson(0.3, n); late60 = rng.poisson(0.1, n); late90 = rng.poisson(0.15, n)
    income = np.where(rng.random(n) < 0.2, np.nan, rng.lognormal(8.6, 0.6, n)).round(0)
    debt = np.clip(rng.lognormal(-1.2, 0.9, n), 0, 50)
    logit = -3.2 + 2.2 * util - 0.02 * (age - 45) + 0.8 * late30 + 1.1 * late60 + 1.4 * late90 + 0.15 * debt
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"SeriousDlqin2yrs": y, "RevolvingUtilizationOfUnsecuredLines": util, "age": age, "NumberOfTime30-59DaysPastDueNotWorse": late30,
                         "DebtRatio": debt, "MonthlyIncome": income, "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n), "NumberOfTimes90DaysLate": late90,
                         "NumberRealEstateLoansOrLines": rng.integers(0, 4, n), "NumberOfTime60-89DaysPastDueNotWorse": late60, "NumberOfDependents": rng.integers(0, 5, n)})
