"""End-to-end behavioral-causal study.

Pipeline:
  1. simulate 300 episodes of a heterogeneous fleet under baseline / spoofed /
     suppressed / calibrated-uncertainty advisory conditions;
  2. decode the latent trust state from kinematic telemetry with the Gaussian HMM;
  3. build a *confounded* observational sample (treatment assignment depends on
     driver class) from the simulator's potential outcomes, and estimate the
     spoofing -> safety effect with three methods:
        - Static Compliance Rule (assumes compliance invariant -> blind to trust),
        - Standard correlational estimate (naive, confounded),
        - Causal AIPW + SCM mediation (ours);
  4. decompose spoofed vs. suppressed harm into Natural Direct / Indirect Effects;
  5. evaluate the calibrated-uncertainty counterfactual policy.

Writes results/*.csv consumed by visualization/make_pub_figures.py.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sim.heterogeneous_driver_sim import (REGIMES, simulate_study)
from models.trust_hmm import GaussianHMM, align_states, decoding_accuracy
from models.causal_mediation import (aipw_ate, mediation_effects, naive_ate)

RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)

# class-dependent treatment propensity -> induces confounding in the
# observational sample (aggressive drivers over-exposed to the spoofing arm)
_PROPENSITY = {"aggressive": 0.75, "cautious": 0.30, "cav_assisted": 0.50}


def build_observational(po: pd.DataFrame, arm: str, seed=7):
    """Synthesize a confounded observational dataset from potential outcomes.

    arm in {"spoofed","suppressed"} selects the treated potential outcome; the
    control is always baseline. T is drawn from a class-dependent propensity, so
    the naive contrast is confounded by driver class while the true ATE is known.
    """
    rng = np.random.default_rng(seed)
    e = po["driver_class"].map(_PROPENSITY).to_numpy()
    T = (rng.random(len(po)) < e).astype(int)
    suffix = "supp" if arm == "suppressed" else "1"
    d1, t1, c1 = f"drac_{suffix}" if arm == "suppressed" else "drac1", \
        f"trust_{suffix}" if arm == "suppressed" else "trust1", \
        f"critical_{suffix}" if arm == "suppressed" else "critical1"
    drac = np.where(T == 1, po[d1], po["drac0"])
    trust = np.where(T == 1, po[t1], po["trust0"])
    crit = np.where(T == 1, po[c1], po["critical0"])
    df = pd.DataFrame({"T": T, "driver_class": po["driver_class"].values,
                       "trust": trust, "peak_drac": drac, "critical": crit})
    # one-hot class covariates
    for c in ["aggressive", "cautious", "cav_assisted"]:
        df[f"cls_{c}"] = (df["driver_class"] == c).astype(int)
    true_ate_drac = float((po[d1] - po["drac0"]).mean())
    true_ate_crit = float((po[c1] - po["critical0"]).mean())
    return df, {"drac": true_ate_drac, "critical": true_ate_crit}


def main():
    print("[1/5] simulating 300-episode behavioral study ...")
    out = simulate_study(n_episodes=300, drivers_per_episode=12, seed=20260101)
    rec, po = out.records, out.potential_outcomes
    rec.to_csv(os.path.join(RESULTS, "safety_surrogates.csv"), index=False)

    print("[2/5] decoding latent trust states (Gaussian HMM / Baum-Welch) ...")
    train = [s["obs"] for s in out.sequences[:600]]
    hmm = GaussianHMM(n_states=4, n_iter=40, seed=0).fit(train)
    acc = decoding_accuracy(hmm, out.sequences[:600])
    # align learned states to regime labels for interpretable outputs
    preds = np.concatenate([hmm.decode(s["obs"]) for s in out.sequences[:600]])
    truth = np.concatenate([s["states"] for s in out.sequences[:600]])
    mapping = align_states(preds, truth, 4)          # pred_state -> true regime
    inv = np.argsort(mapping)                         # true regime -> pred_state
    print(f"      HMM decoding accuracy = {acc:.3f}")

    # transition matrix re-ordered into regime order
    A = hmm.A[np.ix_(inv, inv)]
    pd.DataFrame(A, index=REGIMES, columns=REGIMES).to_csv(
        os.path.join(RESULTS, "hmm_transitions.csv"))

    # state occupancy by condition (decoded, relabeled to regimes)
    occ_rows = []
    for cond in ["baseline", "spoofed", "suppressed", "calibrated"]:
        seqs = [s for s in out.sequences if s["condition"] == cond]
        counts = np.zeros(4)
        for s in seqs:
            dec = mapping[hmm.decode(s["obs"])]       # -> regime index
            for k in range(4):
                counts[k] += (dec == k).sum()
        counts /= counts.sum()
        occ_rows.append({"condition": cond,
                         **{REGIMES[k]: counts[k] for k in range(4)}})
    pd.DataFrame(occ_rows).to_csv(
        os.path.join(RESULTS, "hmm_occupancy.csv"), index=False)

    print("[3/5] estimating spoofing -> safety effect (3 methods) ...")
    eff_rows = []
    for arm in ["spoofed", "suppressed"]:
        df, true_ate = build_observational(po, arm)
        cov = ["cls_aggressive", "cls_cav_assisted"]  # cautious = reference
        naive = naive_ate(df, "T", "peak_drac")
        aipw = aipw_ate(df, "T", "peak_drac", cov, n_boot=300)
        # Static Compliance Rule: assumes compliance invariant to trust -> the
        # trust-mediated channel is structurally invisible; its predicted effect
        # is the complier-only contrast generalized to all drivers (~no change).
        compliers = df[df["trust"] > 0.6]
        static = naive_ate(compliers, "T", "peak_drac") \
            if compliers["T"].nunique() == 2 else 0.0
        for method, est in [("Static Compliance Rule", static),
                            ("Standard Correlational (naive)", naive),
                            ("Causal AIPW+SCM (Ours)", aipw["ate"])]:
            eff_rows.append({
                "contrast": f"{arm}_vs_baseline", "method": method,
                "outcome": "peak_drac", "estimate": est,
                "true_ate": true_ate["drac"],
                "abs_bias": abs(est - true_ate["drac"]),
                "ci_low": aipw["ci_low"] if "Ours" in method else np.nan,
                "ci_high": aipw["ci_high"] if "Ours" in method else np.nan,
            })

    print("[4/5] causal mediation: trust-mediated (NIE) vs direct (NDE) ...")
    med_rows = []
    for arm in ["spoofed", "suppressed"]:
        df, _ = build_observational(po, arm)
        cov = ["cls_aggressive", "cls_cav_assisted"]
        # overall
        for cls in ["ALL", "aggressive", "cautious", "cav_assisted"]:
            sub = df if cls == "ALL" else df[df["driver_class"] == cls]
            use_cov = cov if cls == "ALL" else []
            if sub["T"].nunique() < 2 or len(sub) < 40:
                continue
            m = mediation_effects(sub, "T", "trust", "peak_drac",
                                  use_cov, n_boot=300)
            med_rows.append({
                "contrast": f"{arm}_vs_baseline", "driver_class": cls,
                "NDE": m["NDE"], "NIE": m["NIE"], "ATE": m["ATE"],
                "prop_mediated": m["prop_mediated"],
                "NDE_lo": m["NDE_ci"][0], "NDE_hi": m["NDE_ci"][1],
                "NIE_lo": m["NIE_ci"][0], "NIE_hi": m["NIE_ci"][1],
                "ATE_lo": m["ATE_ci"][0], "ATE_hi": m["ATE_ci"][1],
            })
    pd.DataFrame(med_rows).to_csv(
        os.path.join(RESULTS, "causal_mediation.csv"), index=False)

    pd.DataFrame(eff_rows).to_csv(
        os.path.join(RESULTS, "causal_effects.csv"), index=False)

    print("[5/5] counterfactual policy (calibrated-uncertainty mitigation) ...")
    pol = (rec.groupby("condition")
           .agg(mean_drac=("peak_drac", "mean"),
                mean_ttc=("min_ttc", "mean"),
                critical_rate=("critical_ttc", "mean"),
                mean_trust=("final_trust", "mean"),
                heeded_rate=("heeded_warning", "mean"))
           .reindex(["baseline", "spoofed", "suppressed", "calibrated"])
           .reset_index())
    pol.to_csv(os.path.join(RESULTS, "policy_counterfactual.csv"), index=False)

    # trust trajectories (mean over spoofed drivers, by class) for fig1
    tr_rows = []
    for cls in ["aggressive", "cautious", "cav_assisted"]:
        seqs = [s for s in out.sequences
                if s["condition"] == "spoofed" and s["driver_class"] == cls]
        L = min(len(s["trust_trace"]) for s in seqs)
        mat = np.array([s["trust_trace"][:L] for s in seqs])
        mean, sd = mat.mean(0), mat.std(0)
        for i in range(L):
            tr_rows.append({"driver_class": cls, "encounter": i,
                            "trust": mean[i], "trust_sd": sd[i]})
    pd.DataFrame(tr_rows).to_csv(
        os.path.join(RESULTS, "trust_trajectories.csv"), index=False)

    # ---- console summary ----
    pd.set_option("display.width", 170)
    print("\n================ SPOOFING -> SAFETY EFFECT (peak DRAC) ============")
    ef = pd.DataFrame(eff_rows)
    print(ef[ef.contrast == "spoofed_vs_baseline"][
        ["method", "estimate", "true_ate", "abs_bias"]].round(3).to_string(
        index=False))
    print("\n================ MEDIATION DECOMPOSITION (ALL drivers) ===========")
    md = pd.DataFrame(med_rows)
    print(md[md.driver_class == "ALL"][
        ["contrast", "NDE", "NIE", "ATE", "prop_mediated"]].round(3).to_string(
        index=False))
    print("\n================ COUNTERFACTUAL POLICY ===========================")
    print(pol.round(3).to_string(index=False))
    print(f"\nHMM decoding accuracy: {acc:.3f}")
    print(f"Wrote results to {RESULTS}")


if __name__ == "__main__":
    main()
