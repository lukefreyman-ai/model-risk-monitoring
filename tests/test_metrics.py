import numpy as np
import pandas as pd

from mrm import metrics as M


def test_psi_zero_for_same_distribution_and_positive_for_shift():
    rng = np.random.default_rng(0); a = rng.normal(600, 30, 20000); b = rng.normal(600, 30, 20000); c = rng.normal(585, 30, 20000)
    same, _ = M.psi(a, b); shifted, tab = M.psi(a, c)
    assert same < 0.01 and shifted > 0.1 and len(tab) == 10


def test_csi_on_categorical_bins():
    e = pd.Series(pd.Categorical(["a"] * 70 + ["b"] * 30)); a = pd.Series(pd.Categorical(["a"] * 40 + ["b"] * 60, categories=["a", "b"]))
    v, tab = M.csi(e, a)
    assert v > 0.1 and list(tab.bin) == ["a", "b"]


def test_ks_gini_calibration():
    rng = np.random.default_rng(1); y = (rng.random(5000) < 0.1).astype(int); p = np.clip(0.1 + 0.4 * y + rng.normal(0, 0.15, 5000), 0.001, 0.999)
    assert M.ks_stat(y, p) > 0.5 and M.gini(y, p) > 0.5
    cal = M.calibration_table(y, p); assert 5 <= len(cal) <= 10 and abs(cal.n.sum() - 5000) == 0
    chi2, pv = M.hosmer_lemeshow(y, p); assert chi2 >= 0 and 0 <= pv <= 1
    assert M.rag(0.05, 0.1, 0.25) == "GREEN" and M.rag(0.3, 0.1, 0.25) == "RED"
