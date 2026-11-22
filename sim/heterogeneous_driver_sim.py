"""Heterogeneous multi-lane corridor simulator with latent driver trust dynamics
under spoofed / suppressed / calibrated connected-corridor advisories.

The simulator models asymmetric trust collapse (fast) and recovery (slow) -> a
*compliance hysteresis* loop. Each driver traverses a sequence of advisory
encounters (an early attack window, a recovery window, and a terminal legitimate
hazard) while emitting kinematic telemetry

    [reaction_time (s), time_headway (s), decel_jerk (m/s^3), lane_change_freq]

and, at the terminal hazard, surrogate safety measures

    peak DRAC (deceleration rate to avoid a crash, m/s^2) and
    min TTC (time-to-collision, s); critical exposure = TTC < 1.5 s.

Because it is a simulator we can draw *both* potential outcomes (safety under
baseline and under a counterfactual condition) for the same driver, giving a
ground-truth Average Treatment Effect to validate the causal estimators against.

NOTE: This is a synthetic generative model, not field data. No public dataset of
driver trust collapse under spoofed V2X advisories exists; absolute rates are
modeling assumptions. The contribution is the *relative* comparison of estimators
under a shared, known data-generating process. Honestly flagged in the README.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Latent behavioral regimes (ground-truth labels for HMM evaluation)
# ----------------------------------------------------------------------------
REGIMES = ["Calibrated_Compliance", "Skeptical_Delay", "Herd_Following",
           "Advisory_Ignore"]
REGIME_IDX = {r: i for i, r in enumerate(REGIMES)}

# Gaussian emission means per regime over the 4 telemetry channels.
#            [reaction_time, time_headway, decel_jerk, lane_change_freq]
_EMISSION_MEAN = {
    "Calibrated_Compliance": np.array([0.90, 1.85, 2.0, 0.20]),
    "Skeptical_Delay":       np.array([1.65, 1.45, 3.6, 0.45]),
    "Herd_Following":        np.array([1.30, 1.05, 3.0, 1.15]),
    "Advisory_Ignore":       np.array([2.25, 0.80, 5.1, 0.30]),
}
# Emission spreads are wide enough that adjacent regimes overlap (real driver
# telemetry does not cleanly separate), so the HMM recovers states imperfectly.
_EMISSION_STD = {
    "Calibrated_Compliance": np.array([0.26, 0.36, 0.85, 0.15]),
    "Skeptical_Delay":       np.array([0.36, 0.34, 1.05, 0.22]),
    "Herd_Following":        np.array([0.33, 0.30, 0.95, 0.33]),
    "Advisory_Ignore":       np.array([0.42, 0.27, 1.35, 0.20]),
}

CONDITIONS = ["baseline", "spoofed", "suppressed", "calibrated"]

# ----------------------------------------------------------------------------
# Heterogeneous driver classes
# ----------------------------------------------------------------------------


@dataclass
class DriverClass:
    name: str
    trust0: float          # initial (calibrated) trust
    theta: float           # compliance threshold (lower => complies more easily)
    decay: float           # multiplicative trust loss on a detected false advisory
    recover: float         # additive trust gain on a confirmed true advisory
    base_react: float      # nominal perception-reaction time (s)
    speed: float           # free-flow approach speed (m/s)
    cav_assist: float      # 0..1 fraction of reaction/ DRAC mitigated by automation


DRIVER_CLASSES = {
    "aggressive": DriverClass("aggressive", trust0=0.70, theta=0.55, decay=0.45,
                              recover=0.05, base_react=0.9, speed=31.0,
                              cav_assist=0.0),
    "cautious":   DriverClass("cautious", trust0=0.85, theta=0.40, decay=0.60,
                              recover=0.08, base_react=1.1, speed=26.0,
                              cav_assist=0.0),
    "cav_assisted": DriverClass("cav_assisted", trust0=0.80, theta=0.45,
                                decay=0.55, recover=0.07, base_react=1.0,
                                speed=28.0, cav_assist=0.55),
}


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def _regime_of(trust: float, density: float) -> str:
    """Map latent trust + local traffic density to a behavioral regime."""
    if trust > 0.65:
        return "Calibrated_Compliance"
    if trust > 0.42:
        return "Skeptical_Delay"
    # low trust: herd-follow in dense traffic, otherwise ignore advisories
    return "Herd_Following" if density > 0.6 else "Advisory_Ignore"


# regimes form a chain ordered by trust; a driver near a boundary can *look*
# like an adjacent regime on a given encounter (irreducible behavioral ambiguity)
_NEIGHBORS = {
    "Calibrated_Compliance": ["Skeptical_Delay"],
    "Skeptical_Delay":       ["Calibrated_Compliance", "Herd_Following"],
    "Herd_Following":        ["Skeptical_Delay", "Advisory_Ignore"],
    "Advisory_Ignore":       ["Herd_Following"],
}


def _emit(regime: str, rng: np.random.Generator, blur: float = 0.18) -> np.ndarray:
    """Sample telemetry for ``regime``; with prob ``blur`` the *appearance*
    matches an adjacent regime (the ground-truth label is unchanged)."""
    emit_regime = regime
    if rng.random() < blur:
        emit_regime = rng.choice(_NEIGHBORS[regime])
    mu, sd = _EMISSION_MEAN[emit_regime], _EMISSION_STD[emit_regime]
    obs = rng.normal(mu, sd)
    obs[0] = max(obs[0], 0.25)   # reaction time
    obs[1] = max(obs[1], 0.35)   # headway
    obs[2] = max(obs[2], 0.3)    # jerk
    obs[3] = max(obs[3], 0.0)    # lane-change frequency
    return obs


# ----------------------------------------------------------------------------
# Trust / safety dynamics for one driver traversing the corridor
# ----------------------------------------------------------------------------


def _run_driver(dc: DriverClass, condition: str, rng: np.random.Generator,
                n_pre: int = 5):
    """Simulate one driver: attack window -> recovery -> terminal legit hazard.

    Returns (obs_seq [T,4], state_seq [T], final_trust, safety dict).
    """
    trust = float(np.clip(rng.normal(dc.trust0, 0.05), 0.05, 0.97))
    obs_seq, state_seq, trust_trace = [], [], []

    for t in range(n_pre):
        trust_trace.append(trust)
        density = float(np.clip(rng.beta(2, 3) + (0.15 if condition == "spoofed"
                                                   else 0.0), 0, 1))
        regime = _regime_of(trust, density)
        obs_seq.append(_emit(regime, rng))
        state_seq.append(REGIME_IDX[regime])

        p_comply = _sigmoid(6.0 * (trust - dc.theta))
        complied = rng.random() < p_comply

        if condition == "baseline":
            # advisories are legitimate -> reinforce trust when complied
            if complied:
                trust = min(0.97, trust + dc.recover)
        elif condition == "spoofed":
            # false advisory; complying drivers feel betrayed -> sharp drop
            if complied:
                trust = max(0.05, trust * dc.decay)
            else:
                trust = max(0.05, trust - 0.02)
        elif condition == "suppressed":
            # real minor hazard, no advisory delivered -> mild erosion of
            # reliance (system felt absent) but no betrayal spike
            trust = max(0.05, trust - 0.03)
        elif condition == "calibrated":
            # uncertainty-tagged advisory that proves false: trust updates, but
            # the calibrated uncertainty blunts the betrayal (asymmetry damped)
            if complied:
                damped = dc.decay + (1 - dc.decay) * 0.55
                trust = max(0.05, trust * damped)

    # ---- recovery window (attack has stopped; advisories legitimate again) ---
    for _ in range(3):
        trust_trace.append(trust)
        density = float(np.clip(rng.beta(2, 3), 0, 1))
        regime = _regime_of(trust, density)
        obs_seq.append(_emit(regime, rng))
        state_seq.append(REGIME_IDX[regime])
        p_comply = _sigmoid(6.0 * (trust - dc.theta))
        if rng.random() < p_comply:
            trust = min(0.97, trust + dc.recover)   # slow additive recovery

    final_trust = trust
    trust_trace.append(final_trust)

    # ---- terminal LEGITIMATE hazard: measure surrogate safety ----
    density = float(np.clip(rng.beta(2, 3), 0, 1))
    regime = _regime_of(final_trust, density)
    obs_seq.append(_emit(regime, rng))
    state_seq.append(REGIME_IDX[regime])

    safety = _terminal_hazard(dc, final_trust, condition, rng)

    return (np.array(obs_seq), np.array(state_seq, dtype=int), final_trust,
            safety, np.array(trust_trace))


def _terminal_hazard(dc: DriverClass, trust: float, condition: str,
                     rng: np.random.Generator) -> dict:
    """Surrogate safety at a real downstream hazard (e.g., a stopped queue).

    A driver travelling at speed ``v`` must stop before a hazard first visible at
    sight distance ``D_sight``. A *heeded* legitimate warning adds advance-notice
    distance (the driver begins slowing earlier); an ignored or suppressed
    warning leaves only line-of-sight. DRAC is the deceleration required to stop
    in the remaining distance; TTC < 1.5 s is a critical conflict.
    """
    v = max(5.0, rng.normal(dc.speed, 2.0))
    d_sight = max(40.0, rng.normal(95.0, 15.0))       # line-of-sight to hazard (m)

    warning_present = condition != "suppressed"
    p_comply = _sigmoid(6.0 * (trust - dc.theta))
    heeds = warning_present and (rng.random() < p_comply)

    # advance-notice headroom: a heeded warning (or on-board sensing) lets the
    # driver start slowing well upstream of line-of-sight
    advance = 0.0
    if heeds:
        advance += max(0.0, rng.normal(120.0, 20.0))
    advance += 60.0 * dc.cav_assist                   # CAV sensing sees it early
    detect = d_sight + advance

    # effective perception-reaction time
    react = dc.base_react + rng.normal(0, 0.1)
    if not heeds:
        react += rng.normal(1.1, 0.25)                # late reaction (hysteresis)
    react = max(0.3, react * (1.0 - 0.6 * dc.cav_assist))

    # small DIRECT effect of an active spoofing campaign (elevated approach var)
    if condition == "spoofed":
        v += rng.normal(0.8, 0.5)

    remaining = detect - v * react                     # braking distance left (m)
    remaining_eff = max(remaining, 1.0)
    drac = v ** 2 / (2.0 * remaining_eff)              # required decel (m/s^2)
    drac *= (1.0 - 0.5 * dc.cav_assist)                # AEB caps required decel
    drac = float(min(drac, 12.0))                      # tyre-friction ceiling
    min_ttc = float(max(remaining, 0.0) / v)           # time margin (s)

    return {
        "approach_speed": v, "reaction_time_hazard": react,
        "peak_drac": drac, "min_ttc": min_ttc,
        "critical_ttc": int(min_ttc < 1.5),
        "heeded_warning": int(heeds),
    }


# ----------------------------------------------------------------------------
# Study driver: 300 episodes x heterogeneous fleet
# ----------------------------------------------------------------------------


@dataclass
class StudyOutput:
    records: pd.DataFrame                   # one row per driver-episode
    sequences: list = field(default_factory=list)   # telemetry seqs for the HMM
    potential_outcomes: pd.DataFrame = None  # ground-truth counterfactuals


def simulate_study(n_episodes: int = 300, drivers_per_episode: int = 12,
                   seed: int = 20260101) -> StudyOutput:
    """Run the full behavioral study.

    Conditions are assigned per episode (balanced across the 4 conditions). For
    every driver we ALSO draw the counterfactual pair (baseline vs spoofed) with
    a frozen driver seed, yielding ground-truth potential outcomes Y0, Y1.
    """
    rng = np.random.default_rng(seed)
    class_names = list(DRIVER_CLASSES)

    rows, sequences, po_rows = [], [], []
    for ep in range(n_episodes):
        condition = CONDITIONS[ep % len(CONDITIONS)]
        for d in range(drivers_per_episode):
            cname = class_names[(ep * drivers_per_episode + d) % len(class_names)]
            dc = DRIVER_CLASSES[cname]
            did = f"EP{ep:03d}-D{d:02d}"

            # factual run under this episode's assigned condition
            dseed = int(rng.integers(0, 2**31 - 1))
            obs, states, ftrust, safety, trace = _run_driver(
                dc, condition, np.random.default_rng(dseed))
            sequences.append({"obs": obs, "states": states,
                              "driver_class": cname, "condition": condition,
                              "driver_id": did, "trust_trace": trace})
            rows.append({
                "episode": ep, "driver_id": did, "driver_class": cname,
                "condition": condition, "final_trust": ftrust,
                "mean_reaction": float(obs[:, 0].mean()),
                "mean_headway": float(obs[:, 1].mean()),
                "mean_jerk": float(obs[:, 2].mean()),
                "mean_lane_change": float(obs[:, 3].mean()),
                **safety,
            })

            # ground-truth potential outcomes: same driver + seed, all arms
            _, _, tr0, s0, _ = _run_driver(dc, "baseline",
                                           np.random.default_rng(dseed))
            _, _, tr1, s1, _ = _run_driver(dc, "spoofed",
                                           np.random.default_rng(dseed))
            _, _, trs, ss, _ = _run_driver(dc, "suppressed",
                                           np.random.default_rng(dseed))
            po_rows.append({
                "driver_id": did, "driver_class": cname,
                "trust0": tr0, "trust1": tr1, "trust_supp": trs,
                "drac0": s0["peak_drac"], "drac1": s1["peak_drac"],
                "drac_supp": ss["peak_drac"],
                "critical0": s0["critical_ttc"], "critical1": s1["critical_ttc"],
                "critical_supp": ss["critical_ttc"],
            })

    return StudyOutput(records=pd.DataFrame(rows), sequences=sequences,
                       potential_outcomes=pd.DataFrame(po_rows))


if __name__ == "__main__":
    out = simulate_study(n_episodes=60)
    print(out.records.groupby("condition")[["peak_drac", "min_ttc",
                                             "critical_ttc", "final_trust"]]
          .mean().round(3))
    po = out.potential_outcomes
    print("\nGround-truth ATE (spoofed-baseline):")
    print("  peak DRAC  :", round((po.drac1 - po.drac0).mean(), 3), "m/s^2")
    print("  critical TTC:", round((po.critical1 - po.critical0).mean(), 3))
