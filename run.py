#!/usr/bin/env python3
"""model-risk-monitoring: fit the champion scorecard on the development window, monitor it (and a challenger) across
simulated windows, and write an SR 11-7-style validation report.

  python run.py                 uses data/cs-training.csv (Give Me Some Credit) -> results/
  python run.py --synthetic     smoke run on synthetic data (never for reported numbers)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mrm import data as D, report as R  # noqa: E402
from mrm.monitor import run_monitoring  # noqa: E402

ROOT = Path(__file__).resolve().parent
PURPOSE = ("Application scorecard predicting the probability that a retail borrower experiences 90+ days delinquency within two years "
           "(target SeriousDlqin2yrs). Intended use: ranking applicants for approve/decline and pricing tiers. Development data: the Kaggle "
           "'Give Me Some Credit' sample (150,000 borrowers). Monitoring windows are simulated from held-out slices of the same data with a "
           "documented covariate shift (config.yaml -> windows.drift), because the public dataset carries no dates.")
SOUNDNESS = ("Weight-of-evidence binning with monotone risk ordering per characteristic and a logistic regression on the WOE-transformed "
             "features; points scaled so that 600 points correspond to 50:1 odds and 20 points double the odds. All characteristics show "
             "information value above 0.02 and variance inflation factors below 2 (see credit-risk-scorecard). Monotonicity of bad rate "
             "across bins is enforced by the binning choices in config.yaml.")
LIMITATIONS = [
    "Monitoring windows are simulated by re-weighting rows of one snapshot; there is no true time dimension, so trend statements describe the injected shift, not real portfolio evolution.",
    "Outcome labels are available for every window immediately; in production a 24-month performance window means KS/Gini lag PSI by two years and early monitoring must rely on stability metrics alone.",
    "Hosmer-Lemeshow p-values are near zero at these sample sizes regardless of practical fit; the absolute calibration gap is the operative test.",
    "The challenger is untuned (fixed depth 4, 300 trees) and shares the development window; its lift is an upper-bound illustration, not a validated alternative.",
    "No override, reject-inference or fair-lending (disparate impact) analysis is included; both would be required for a production validation.",
]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml")); ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config)); out = Path(a.out); t0 = time.time()
    df = D.make_synthetic(30000, seed=int(cfg["data"]["seed"])) if a.synthetic else D.load_gmsc(ROOT / cfg["data"]["path"])
    window = D.assign_windows(df, cfg["windows"] | {"target": cfg["data"]["target"]}, int(cfg["data"]["seed"]))
    df, window = df[window >= 0].reset_index(drop=True), window[window >= 0].reset_index(drop=True)   # thinned-out rows are unused
    labels = cfg["windows"]["labels"]
    print(f"rows={len(df):,} bad rate={df[cfg['data']['target']].mean():.3%} windows={dict(window.value_counts().sort_index())}")
    pack, champion = run_monitoring(df, window, labels, cfg)
    p = pack.performance
    print(p[["window", "n", "bad_rate", "champion_ks", "champion_gini", "challenger_gini", "score_psi", "psi_rag", "avg_pd", "actual_minus_predicted", "calibration_rag"]].round(4).to_string(index=False))
    print("\nCSI (feature x window):"); print(pack.csi.pivot(index="feature", columns="window", values="csi").round(3).to_string())
    out.mkdir(parents=True, exist_ok=True)
    for name, df_ in [("performance", p), ("psi_scores", pack.psi_scores), ("csi", pack.csi), ("calibration", pack.calibration), ("score_bands", pack.bands),
                      ("scorecard_points", pack.points), ("information_value", pack.ivs), ("challenger_importance", pack.challenger_importance)]:
        df_.to_csv(out / f"{name}.csv", index=False)
    path = R.render(pack, out, cfg, "Retail application scorecard (WOE logistic, GMSC)", int((window == 0).sum()), LIMITATIONS, PURPOSE, SOUNDNESS)
    rag, _ = R.overall(pack, cfg["thresholds"])
    meta = {"dataset": "synthetic" if a.synthetic else "GMSC cs-training.csv", "rows": int(len(df)), "windows": {l: int((window == i).sum()) for i, l in enumerate(labels)},
            "overall_rag": rag, "findings": R.findings(pack, cfg["thresholds"]), "dev_gini": float(p.champion_gini.iloc[0]), "dev_ks": float(p.champion_ks.iloc[0]),
            "max_score_psi": float(p.score_psi.max()), "elapsed_seconds": round(time.time() - t0, 1)}
    (out / "metrics.json").write_text(json.dumps(meta, indent=1))
    print(f"\noverall: {rag}"); [print(f"  {f['severity']:6} {f['text']}") for f in meta["findings"]]
    print(f"report: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}  ({meta['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
