"""Turns a MonitoringPack into plots, an HTML report (Jinja2) and a findings list."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader

from mrm.monitor import MonitoringPack

TEMPLATES = Path(__file__).resolve().parents[1] / "templates"


def _html(df: pd.DataFrame, floatfmt: str = "{:.3f}", rag_cols: tuple = ()) -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: floatfmt.format(v) if pd.notna(v) else "")
    html = d.to_html(index=False, border=0, escape=False)
    for rag in ("GREEN", "AMBER", "RED"):
        html = html.replace(f"<td>{rag}</td>", f'<td class="{rag}">{rag}</td>')
    return html


def plots(pack: MonitoringPack, out: Path, thr: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = pack.performance
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(p.window, p.champion_ks, "o-", label="champion KS"); ax[0].plot(p.window, p.challenger_ks, "s--", label="challenger KS")
    ax[0].axhline(thr["ks_min"], color="r", ls=":", label=f"KS floor {thr['ks_min']}"); ax[0].set_title("KS by window"); ax[0].legend(fontsize=8); ax[0].set_ylim(0, 1)
    ax[1].plot(p.window, p.champion_gini, "o-", label="champion Gini"); ax[1].plot(p.window, p.challenger_gini, "s--", label="challenger Gini")
    ax[1].set_title("Gini by window"); ax[1].legend(fontsize=8); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out / "performance.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(pack.psi_scores.window, pack.psi_scores.score_psi, color=["#2a6fdb" if v < thr["psi_green"] else "#e0a800" if v < thr["psi_amber"] else "#d62828" for v in pack.psi_scores.score_psi])
    ax.axhline(thr["psi_green"], color="#e0a800", ls="--", lw=1); ax.axhline(thr["psi_amber"], color="#d62828", ls="--", lw=1)
    ax.set_title("Score PSI vs development sample"); fig.tight_layout(); fig.savefig(out / "psi.png", dpi=130); plt.close(fig)

    piv = pack.bands.pivot(index="window", columns="band", values="share").reindex(p.window)
    ax = piv.plot(kind="bar", stacked=True, figsize=(8, 3.6), title="score band mix by window", colormap="viridis"); ax.legend(fontsize=7, ncol=3)
    ax.figure.tight_layout(); ax.figure.savefig(out / "score_bands.png", dpi=130); plt.close(ax.figure)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    for w, g in pack.calibration.groupby("window", sort=False):
        ax.plot(g.predicted, g.actual, "o-", ms=3, lw=1, label=w)
    lim = max(pack.calibration.actual.max(), pack.calibration.predicted.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k:", lw=1); ax.set_xlabel("predicted PD (decile mean)"); ax.set_ylabel("actual default rate"); ax.set_title("Calibration by window")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "calibration.png", dpi=130); plt.close(fig)


def findings(pack: MonitoringPack, thr: dict) -> list[dict]:
    p = pack.performance; out = []
    worst_psi = p.loc[p.score_psi.idxmax()]
    if worst_psi.score_psi >= thr["psi_amber"]:
        out.append({"severity": "HIGH", "text": f"Score PSI reaches {worst_psi.score_psi:.3f} in {worst_psi.window} (> {thr['psi_amber']}): the scored population has shifted materially from development. Investigate acquisition mix before trusting cut-offs."})
    elif worst_psi.score_psi >= thr["psi_green"]:
        out.append({"severity": "MEDIUM", "text": f"Score PSI reaches {worst_psi.score_psi:.3f} in {worst_psi.window} (amber band {thr['psi_green']}-{thr['psi_amber']}). Monitor; no action required yet."})
    else:
        out.append({"severity": "LOW", "text": f"Score PSI stays below {thr['psi_green']} in every window (max {worst_psi.score_psi:.3f} in {worst_psi.window})."})
    c = pack.csi[pack.csi.rag != "GREEN"]
    if len(c):
        top = c.sort_values("csi", ascending=False).iloc[0]
        out.append({"severity": "MEDIUM", "text": f"Characteristic drift: {len(c)} feature-window pairs breach CSI {thr['csi_amber']}; largest is {top.feature} in {top.window} (CSI {top.csi:.3f}). These characteristics drive the score shift."})
    else:
        out.append({"severity": "LOW", "text": "No characteristic breaches the CSI threshold."})
    lo_ks = p[p.champion_ks < thr["ks_min"]]
    if len(lo_ks):
        out.append({"severity": "HIGH", "text": f"Champion KS falls below {thr['ks_min']} in {', '.join(lo_ks.window)}."})
    else:
        out.append({"severity": "LOW", "text": f"Champion KS stays above {thr['ks_min']} (range {p.champion_ks.min():.3f}-{p.champion_ks.max():.3f}); Gini relative drop vs development at most {p.gini_rel_drop.max()*100:.1f} %."})
    cal = p[p.calibration_rag != "GREEN"]
    if len(cal):
        w = cal.loc[cal.actual_minus_predicted.abs().idxmax()]
        out.append({"severity": "MEDIUM", "text": f"Calibration gap: actual minus predicted default rate is {w.actual_minus_predicted*100:+.2f} pp in {w.window} (> {thr['calibration_max_abs_gap']*100:.0f} pp). Consider an intercept recalibration."})
    else:
        out.append({"severity": "LOW", "text": f"Portfolio-level calibration holds: |actual - predicted| default rate <= {thr['calibration_max_abs_gap']*100:.0f} pp in every window."})
    lift = (p.challenger_gini - p.champion_gini)
    out.append({"severity": "INFO", "text": f"Challenger (gradient boosting) Gini exceeds the champion by {lift.mean()*100:.1f} pts on average ({lift.min()*100:+.1f} to {lift.max()*100:+.1f}) and its relative decay across windows is {p.challenger_gini_rel_drop.max()*100:.1f} % vs {p.gini_rel_drop.max()*100:.1f} % for the champion. Lift alone does not justify replacement: the challenger is not reason-code explainable and would need a full validation cycle."})
    return out


def overall(pack: MonitoringPack, thr: dict) -> tuple[str, str]:
    p = pack.performance
    flags = pd.concat([p.psi_rag, p.ks_rag, p.gini_rag, p.calibration_rag])
    rag = "RED" if (flags == "RED").any() else "AMBER" if (flags == "AMBER").any() else "GREEN"
    text = {"GREEN": "All monitoring metrics are within tolerance; the model remains fit for use.",
            "AMBER": "One or more metrics are in the amber band. The model remains fit for use but the flagged items require investigation and a documented owner response before the next cycle.",
            "RED": "At least one metric breaches its red threshold. Escalate to model owner and Model Risk; restrict use pending remediation."}[rag]
    return rag, text


def render(pack: MonitoringPack, out: Path, cfg: dict, model_name: str, dev_n: int, limitations: list[str], purpose: str, soundness: str) -> Path:
    thr = cfg["thresholds"]
    out.mkdir(parents=True, exist_ok=True)
    plots(pack, out, thr)
    rag, text = overall(pack, thr)
    p = pack.performance
    perf = p[["window", "n", "bad_rate", "champion_ks", "champion_gini", "score_psi", "psi_rag", "ks_rag", "gini_rag", "avg_pd", "actual_minus_predicted", "calibration_rag", "hl_p"]]
    cc = p[["window", "champion_ks", "challenger_ks", "champion_gini", "challenger_gini", "gini_rel_drop", "challenger_gini_rel_drop"]]
    csi_piv = pack.csi.pivot(index="feature", columns="window", values="csi").reindex(columns=p.window).reset_index()
    cal = pack.calibration.pivot(index="decile", columns="window", values="gap").reindex(columns=p.window).reset_index()
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)), autoescape=False)
    html = env.get_template("report.html.j2").render(
        model_name=model_name, generated=time.strftime("%Y-%m-%d %H:%M"), dev_n=f"{dev_n:,}", windows=list(p.window), overall_rag=rag, overall_text=text,
        purpose=purpose, soundness=soundness, thr=thr, iv_table=_html(pack.ivs, "{:.4f}"), points_table=_html(pack.points, "{:.1f}"),
        perf_table=_html(perf), csi_table=_html(csi_piv), cal_table=_html(cal, "{:+.4f}"), cc_table=_html(cc),
        imp_table=_html(pack.challenger_importance.head(10), "{:.3f}"), findings=findings(pack, thr), limitations=limitations,
        challenger_text="The challenger is a gradient-boosted tree model trained on the same development window and raw (unbinned) characteristics. It is used as a yardstick for how much discriminatory power the linear scorecard leaves on the table, not as a replacement candidate.")
    path = out / "validation_report.html"; path.write_text(html)
    return path
