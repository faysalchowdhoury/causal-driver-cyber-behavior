"""Pytest suite: HMM decoding accuracy, causal ATE / mediation recovery on
synthetic ground truth, the spoofing-vs-suppression mechanism contrast, and
figure export."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.heterogeneous_driver_sim import simulate_study, CONDITIONS
from models.trust_hmm import GaussianHMM, decoding_accuracy
from models.causal_mediation import (aipw_ate, ipw_ate, make_synthetic_causal,
                                      mediation_effects, naive_ate)


# ----------------------------- simulator -----------------------------------
def test_simulator_shapes_and_physics():
    out = simulate_study(n_episodes=16, drivers_per_episode=6, seed=1)
    rec = out.records
    assert len(rec) == 16 * 6
    assert set(rec.condition.unique()) <= set(CONDITIONS)
    # physical ranges
    assert rec.peak_drac.between(0, 12.01).all()
    assert (rec.min_ttc >= 0).all()
    assert rec.final_trust.between(0, 1).all()
    # spoofing must erode trust relative to baseline
    tb = rec[rec.condition == "baseline"].final_trust.mean()
    ts = rec[rec.condition == "spoofed"].final_trust.mean()
    assert ts < tb


def test_ground_truth_ate_is_harmful():
    out = simulate_study(n_episodes=40, drivers_per_episode=6, seed=2)
    po = out.potential_outcomes
    # spoofing raises required deceleration (worse safety)
    assert (po.drac1 - po.drac0).mean() > 1.0
    # suppression harms directly while keeping trust ~intact
    assert po.trust_supp.mean() > po.trust1.mean()


# ------------------------------- HMM ---------------------------------------
def test_hmm_decoding_accuracy():
    out = simulate_study(n_episodes=40, drivers_per_episode=8, seed=3)
    seqs = out.sequences[:400]
    hmm = GaussianHMM(n_states=4, n_iter=40, seed=0).fit([s["obs"] for s in seqs])
    acc = decoding_accuracy(hmm, seqs)
    # well above chance (0.25) but not perfect (regimes overlap by construction)
    assert 0.65 < acc < 0.98
    # log-likelihood is monotone non-decreasing under EM
    hist = hmm.loglik_history_
    assert all(b >= a - 1e-6 for a, b in zip(hist, hist[1:]))


# -------------------------- causal estimators ------------------------------
def test_ate_recovery_and_confounding_bias():
    df, truth = make_synthetic_causal(n=6000, seed=5)
    naive = naive_ate(df)
    ipw = ipw_ate(df)
    aipw = aipw_ate(df, n_boot=200)
    # naive is biased by the X-confounding; adjusted estimators recover truth
    assert abs(naive - truth["ATE"]) > 0.6
    assert abs(ipw - truth["ATE"]) < 0.25
    assert abs(aipw["ate"] - truth["ATE"]) < 0.25
    assert aipw["ci_low"] < truth["ATE"] < aipw["ci_high"]


def test_mediation_recovery():
    df, truth = make_synthetic_causal(n=6000, seed=6)
    m = mediation_effects(df, n_boot=200)
    assert abs(m["NDE"] - truth["NDE"]) < 0.3
    assert abs(m["NIE"] - truth["NIE"]) < 0.3
    assert 0.6 < m["prop_mediated"] < 0.85


def test_spoof_is_trust_mediated_suppress_is_direct():
    """Mechanism contrast: spoofing acts through trust (high NIE share),
    suppression acts directly (low NIE share)."""
    out = simulate_study(n_episodes=120, drivers_per_episode=9, seed=7)
    po = out.potential_outcomes
    rng = np.random.default_rng(0)
    prop = {"aggressive": 0.75, "cautious": 0.30, "cav_assisted": 0.50}

    import pandas as pd

    def obs(arm):
        e = po["driver_class"].map(prop).to_numpy()
        T = (rng.random(len(po)) < e).astype(int)
        if arm == "spoofed":
            d, t = po["drac1"], po["trust1"]
        else:
            d, t = po["drac_supp"], po["trust_supp"]
        df = pd.DataFrame({"T": T, "trust": np.where(T == 1, t, po["trust0"]),
                           "peak_drac": np.where(T == 1, d, po["drac0"]),
                           "driver_class": po["driver_class"].values})
        for c in ["aggressive", "cav_assisted"]:   # cautious = reference
            df[f"cls_{c}"] = (df["driver_class"] == c).astype(int)
        return df

    cov = ("cls_aggressive", "cls_cav_assisted")   # adjust for class confounding
    sp = mediation_effects(obs("spoofed"), treat="T", mediator="trust",
                           outcome="peak_drac", covariates=cov, n_boot=150)
    su = mediation_effects(obs("suppressed"), treat="T", mediator="trust",
                           outcome="peak_drac", covariates=cov, n_boot=150)
    assert sp["prop_mediated"] > 0.7      # spoofing: trust-mediated
    assert su["prop_mediated"] < 0.4      # suppression: mostly direct
    assert sp["prop_mediated"] - su["prop_mediated"] > 0.5


# ------------------------------- figures -----------------------------------
def test_figures_from_fixtures(tmp_path, monkeypatch):
    import pandas as pd
    import visualization.make_pub_figures as mpf
    res = tmp_path / "results"; fig = tmp_path / "figures"
    res.mkdir(); fig.mkdir()

    pd.DataFrame([{"driver_class": c, "encounter": i,
                   "trust": 0.8 - 0.07 * i, "trust_sd": 0.05}
                  for c in ["aggressive", "cautious", "cav_assisted"]
                  for i in range(9)]).to_csv(res / "trust_trajectories.csv",
                                             index=False)
    med = []
    for contrast, nde, nie in [("spoofed_vs_baseline", 0.14, 4.2),
                               ("suppressed_vs_baseline", 4.8, 0.4)]:
        for cls in ["ALL", "aggressive", "cautious", "cav_assisted"]:
            med.append({"contrast": contrast, "driver_class": cls,
                        "NDE": nde, "NIE": nie, "ATE": nde + nie,
                        "prop_mediated": nie / (nde + nie),
                        "NDE_lo": nde - 0.5, "NDE_hi": nde + 0.5,
                        "NIE_lo": nie - 0.5, "NIE_hi": nie + 0.5,
                        "ATE_lo": nde + nie - 0.5, "ATE_hi": nde + nie + 0.5})
    pd.DataFrame(med).to_csv(res / "causal_mediation.csv", index=False)

    rng = np.random.default_rng(0)
    rows = []
    for cond, mu in [("baseline", 2.2), ("spoofed", 6.8), ("suppressed", 7.8),
                     ("calibrated", 4.8)]:
        for _ in range(200):
            rows.append({"condition": cond, "driver_class": "aggressive",
                         "peak_drac": float(np.clip(rng.normal(mu, 1.5), 0, 12)),
                         "min_ttc": 3.0, "critical_ttc": 0, "final_trust": 0.5,
                         "heeded_warning": 1})
    pd.DataFrame(rows).to_csv(res / "safety_surrogates.csv", index=False)

    regimes = ["Calibrated_Compliance", "Skeptical_Delay", "Herd_Following",
               "Advisory_Ignore"]
    A = np.full((4, 4), 0.1); np.fill_diagonal(A, 0.7)
    pd.DataFrame(A, index=regimes, columns=regimes).to_csv(
        res / "hmm_transitions.csv")
    pd.DataFrame([{"condition": c, **{r: 0.25 for r in regimes}}
                  for c in ["baseline", "spoofed", "suppressed", "calibrated"]]
                 ).to_csv(res / "hmm_occupancy.csv", index=False)

    monkeypatch.setattr(mpf, "RESULTS", str(res))
    monkeypatch.setattr(mpf, "FIGDIR", str(fig))
    mpf.main()
    for name in ("fig1_trust_hysteresis_loop", "fig2_causal_mediation_forest",
                 "fig3_drac_hazard_distributions", "fig4_hmm_state_transitions"):
        for ext in ("svg", "pdf", "jpg"):
            assert (fig / f"{name}.{ext}").exists()
