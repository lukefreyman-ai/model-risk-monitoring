import yaml
from pathlib import Path

from mrm import data as D, report as R
from mrm.monitor import run_monitoring

ROOT = Path(__file__).resolve().parents[1]


def test_scorecard_and_monitoring_on_synthetic(tmp_path):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    df = D.make_synthetic(12000, seed=3); win = D.assign_windows(df, cfg["windows"] | {"target": "SeriousDlqin2yrs"}, 3)
    assert win.min() == -1 and win.max() == cfg["windows"]["n_windows"] - 1 and (win == 0).mean() > 0.25
    df, win = df[win >= 0].reset_index(drop=True), win[win >= 0].reset_index(drop=True)
    pack, champ = run_monitoring(df, win, cfg["windows"]["labels"], cfg)
    p = pack.performance
    assert len(p) == cfg["windows"]["n_windows"] and (p.champion_gini > 0.3).all() and p.score_psi.iloc[0] < 1e-6
    assert set(pack.csi.feature) == {v["name"] for v in cfg["scorecard"]["features"].values()}
    pts = champ.points_table(); assert len(pts) > 20 and pts.points.notna().all()
    path = R.render(pack, tmp_path, cfg, "test model", int((win == 0).sum()), ["l1"], "purpose", "sound")
    assert path.exists() and "Overall assessment" in path.read_text() and (tmp_path / "psi.png").exists()
    assert len(R.findings(pack, cfg["thresholds"])) >= 4
