"""Runs the full monitoring pack over the simulated windows for a champion scorecard and a challenger model."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from mrm import metrics as M
from mrm.scorecard import Scorecard

SCORE_EDGES = [0, 500, 550, 600, 620, 640, 1000]


@dataclass
class MonitoringPack:
    performance: pd.DataFrame          # per window: n, bad rate, KS, Gini, AUC, mean score, avg PD, HL p, RAG flags (champion + challenger)
    psi_scores: pd.DataFrame           # PSI of champion score distribution vs dev, per window
    csi: pd.DataFrame                  # CSI per feature per window
    calibration: pd.DataFrame          # decile calibration per window (champion)
    bands: pd.DataFrame                # score band mix + bad rate per window
    points: pd.DataFrame               # scorecard points table
    ivs: pd.DataFrame                  # IV per feature on dev
    challenger_importance: pd.DataFrame


def fit_challenger(df_dev: pd.DataFrame, target: str, cfg: dict, seed: int) -> tuple[GradientBoostingClassifier, list[str]]:
    feats = [c for c in df_dev.columns if c != target]
    X = df_dev[feats].fillna(-1)
    clf = GradientBoostingClassifier(n_estimators=int(cfg["n_estimators"]), max_depth=int(cfg["max_depth"]), random_state=seed).fit(X, df_dev[target])
    return clf, feats


def run_monitoring(df: pd.DataFrame, window: pd.Series, labels: list[str], cfg: dict) -> tuple[MonitoringPack, Scorecard]:
    target, thr, seed = cfg["data"]["target"], cfg["thresholds"], int(cfg["data"]["seed"])
    sc_cfg = cfg["scorecard"]
    dev = df[window == 0]
    champion = Scorecard(features=sc_cfg["features"], base_score=sc_cfg["base_score"], base_odds=sc_cfg["base_odds"], pdo=sc_cfg["pdo"]).fit(dev, target)
    challenger, ch_feats = fit_challenger(dev, target, cfg["challenger"], seed)
    dev_scores = champion.score(dev); dev_bins = champion.transform(dev)
    dev_gini_c = M.gini(dev[target].to_numpy(), champion.predict_proba(dev))
    dev_gini_ch = M.gini(dev[target].to_numpy(), challenger.predict_proba(dev[ch_feats].fillna(-1))[:, 1])
    perf, psi_rows, csi_rows, cal_rows, band_rows = [], [], [], [], []
    for w, label in enumerate(labels):
        d = df[window == w]; y = d[target].to_numpy()
        p_c = champion.predict_proba(d); s_c = champion.score(d); p_ch = challenger.predict_proba(d[ch_feats].fillna(-1))[:, 1]
        psi_val, _ = M.psi(dev_scores, s_c)
        hl_chi2, hl_p = M.hosmer_lemeshow(y, p_c)
        g_c, g_ch = M.gini(y, p_c), M.gini(y, p_ch)
        perf.append({"window": label, "n": len(d), "bad_rate": y.mean(), "champion_ks": M.ks_stat(y, p_c), "champion_gini": g_c, "champion_auc": (g_c + 1) / 2,
                     "challenger_ks": M.ks_stat(y, p_ch), "challenger_gini": g_ch, "challenger_auc": (g_ch + 1) / 2,
                     "mean_score": float(s_c.mean()), "avg_pd": float(p_c.mean()), "actual_minus_predicted": float(y.mean() - p_c.mean()),
                     "hl_chi2": hl_chi2, "hl_p": hl_p, "score_psi": psi_val,
                     "gini_rel_drop": (dev_gini_c - g_c) / dev_gini_c, "challenger_gini_rel_drop": (dev_gini_ch - g_ch) / dev_gini_ch,
                     "psi_rag": M.rag(psi_val, thr["psi_green"], thr["psi_amber"]),
                     "ks_rag": "GREEN" if M.ks_stat(y, p_c) >= thr["ks_min"] else "RED",
                     "gini_rag": "GREEN" if (dev_gini_c - g_c) / dev_gini_c < thr["gini_drop_amber"] else "AMBER",
                     "calibration_rag": "GREEN" if abs(y.mean() - p_c.mean()) <= thr["calibration_max_abs_gap"] else "AMBER"})
        psi_rows.append({"window": label, "score_psi": psi_val})
        bins = champion.transform(d)
        for spec in sc_cfg["features"].values():
            v, _ = M.csi(dev_bins[f"{spec['name']}_bin"], bins[f"{spec['name']}_bin"])
            csi_rows.append({"window": label, "feature": spec["name"], "csi": v, "rag": M.rag(v, thr["csi_amber"], thr["csi_amber"] * 2.5)})
        cal = M.calibration_table(y, p_c); cal.insert(0, "window", label); cal_rows.append(cal)
        b = M.score_bands(s_c, y, SCORE_EDGES); b.insert(0, "window", label); band_rows.append(b)
    imp = pd.DataFrame({"feature": ch_feats, "importance": challenger.feature_importances_}).sort_values("importance", ascending=False)
    pack = MonitoringPack(pd.DataFrame(perf), pd.DataFrame(psi_rows), pd.DataFrame(csi_rows), pd.concat(cal_rows, ignore_index=True),
                          pd.concat(band_rows, ignore_index=True), champion.points_table(),
                          pd.DataFrame({"feature": list(champion.ivs), "iv": list(champion.ivs.values())}).sort_values("iv", ascending=False), imp)
    return pack, champion
