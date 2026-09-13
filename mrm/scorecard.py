"""WOE / IV binning and a logistic scorecard - the same recipe as credit-risk-scorecard, packaged as functions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm


def bin_feature(s: pd.Series, bins: list[float], fill: float | None = None) -> pd.Series:
    """Cut a raw characteristic into scorecard bins. Edges are coerced to float (YAML reads `1e9` as a string)."""
    edges = [float(b) for b in bins]
    s = s.fillna(fill) if fill is not None else s
    return pd.cut(s, bins=edges, include_lowest=True)


def woe_table(binned: pd.Series, y: pd.Series) -> pd.DataFrame:
    g = pd.DataFrame({"bin": binned, "y": y}).groupby("bin", observed=False)["y"].agg(["count", "sum"]).rename(columns={"count": "total", "sum": "bad"})
    g["good"] = g["total"] - g["bad"]
    g["bad_pct"] = (g["bad"] + 0.5) / (g["bad"].sum() + 0.5)         # Laplace smoothing keeps empty bins finite
    g["good_pct"] = (g["good"] + 0.5) / (g["good"].sum() + 0.5)
    g["woe"] = np.log(g["good_pct"] / g["bad_pct"])
    g["iv"] = (g["good_pct"] - g["bad_pct"]) * g["woe"]
    g["bad_rate"] = g["bad"] / g["total"].replace(0, np.nan)
    return g


@dataclass
class Scorecard:
    features: dict                      # raw column -> {bins, name, fill}
    woe_maps: dict = field(default_factory=dict)
    ivs: dict = field(default_factory=dict)
    model: object = None
    base_score: int = 600
    base_odds: int = 50
    pdo: int = 20

    @property
    def woe_cols(self) -> list[str]:
        return [f"{v['name']}_woe" for v in self.features.values()]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for col, spec in self.features.items():
            b = bin_feature(df[col], spec["bins"], spec.get("fill"))
            out[f"{spec['name']}_bin"] = b
            out[f"{spec['name']}_woe"] = b.map(self.woe_maps[spec["name"]]).astype(float).fillna(0.0)
        return out

    def fit(self, df: pd.DataFrame, target: str) -> "Scorecard":
        y = df[target]
        for col, spec in self.features.items():
            b = bin_feature(df[col], spec["bins"], spec.get("fill"))
            t = woe_table(b, y)
            self.woe_maps[spec["name"]] = t["woe"].to_dict(); self.ivs[spec["name"]] = float(t["iv"].sum())
        X = sm.add_constant(self.transform(df)[self.woe_cols])
        self.model = sm.Logit(y, X).fit(disp=0)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(self.transform(df)[self.woe_cols], has_constant="add")
        return np.asarray(self.model.predict(X))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Points: base_score at base_odds, pdo points to double the odds (standard scaling)."""
        factor = self.pdo / np.log(2); offset = self.base_score - factor * np.log(self.base_odds)
        X = sm.add_constant(self.transform(df)[self.woe_cols], has_constant="add")
        log_odds_bad = np.asarray(X @ self.model.params)        # logit of P(bad)
        return offset - factor * log_odds_bad                    # higher score = lower risk

    def points_table(self) -> pd.DataFrame:
        factor = self.pdo / np.log(2); offset = self.base_score - factor * np.log(self.base_odds)
        n = len(self.features); rows = []
        for spec in self.features.values():
            coef = self.model.params[f"{spec['name']}_woe"]
            for b, w in self.woe_maps[spec["name"]].items():
                rows.append({"feature": spec["name"], "bin": str(b), "woe": round(w, 4),
                             "points": round(-(coef * w + self.model.params["const"] / n) * factor + offset / n, 1)})
        return pd.DataFrame(rows)
