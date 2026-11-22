# Causal Mechanisms of Driver Compliance Hysteresis and Rear-End Crash Risk Under Spoofed Connected-Corridor Advisories

A reproducible, fully offline research pipeline that separates **why** spoofed or
suppressed connected-vehicle advisories degrade safety, not just **whether** they
do. When Dynamic Message Signs or in-vehicle V2X warnings are **spoofed** (false
work-zone / false speed drop) or **suppressed** during a real hazard, drivers
exhibit an asymmetric **trust collapse and compliance hysteresis**: they discount
later *legitimate* warnings long after the cyber event is over. The headline
result is that two attacks with nearly identical surrogate-safety cost act through
**opposite causal mechanisms** — a distinction correlational models cannot make.

> **Scope & honesty.** No public dataset of driver trust collapse under spoofed
> V2X advisories exists, so this is a **simulation study**: a microscopic
> heterogeneous-driver generator with explicit trust dynamics. Absolute rates are
> modeling assumptions; the contribution is the *relative* behavior of estimators
> under a shared, **known** data-generating process whose ground-truth potential
> outcomes we can draw directly. Results include the honest cases (an imperfect
> HMM decode, a correlational estimate that is biased but not absurd, a mitigation
> that only *partially* recovers safety) rather than a clean sweep.

## Framework

1. **Heterogeneous driver simulator** (`sim/heterogeneous_driver_sim.py`) —
   aggressive / cautious / CAV-assisted drivers with asymmetric trust decay
   (fast on a detected false advisory) and recovery (slow, additive). Emits
   4-channel kinematic telemetry `[reaction time, time headway, decel jerk,
   lane-change freq]` and, at a terminal legitimate hazard, surrogate safety
   measures (**peak DRAC**, **min TTC**; critical = TTC < 1.5 s). Because it is a
   simulator it yields **both** potential outcomes per driver → a ground-truth ATE.
2. **Latent trust-state HMM** (`models/trust_hmm.py`) — a hand-rolled Gaussian
   Hidden Markov Model (Baum-Welch EM, log-space forward-backward, Viterbi) that
   decodes the four latent regimes (`Calibrated_Compliance`, `Skeptical_Delay`,
   `Herd_Following`, `Advisory_Ignore`) from telemetry alone.
3. **SCM + causal mediation** (`models/causal_mediation.py`) — IPW and
   **doubly-robust AIPW** ATE estimators, plus a regression-based decomposition of
   the total effect into a **Natural Direct Effect (NDE)** and a trust-mediated
   **Natural Indirect Effect (NIE)** via the parametric g-formula, with bootstrap
   CIs.

**Baselines:** (1) `Static Compliance Rule` (assumes compliance invariant to
trust — structurally blind to the mediated channel); (2) `Standard Correlational`
(naive, confounded association); (3) `Causal AIPW + SCM Mediation (Ours)`.

## Key results (300 episodes, heterogeneous fleet)

**Recovering the spoofing → safety effect** (peak DRAC, spoofed vs. baseline;
ground-truth ATE = **4.58 m/s²**):

| Method | Estimated ATE (m/s²) | Abs. bias |
|---|---:|---:|
| Static Compliance Rule | 0.20 | 4.39 |
| Standard Correlational (naive) | 5.26 | 0.67 |
| **Causal AIPW + SCM (Ours)** | **4.60**  [4.41, 4.81] | **0.02** |

**Mechanism decomposition** — the same analysis on two attacks:

| Contrast | NDE (direct) | NIE (trust-mediated) | Proportion mediated |
|---|---:|---:|---:|
| **Spoofed** vs. baseline | 0.14 | **4.22** | **96.8%** |
| **Suppressed** vs. baseline | **4.78** | 0.40 | 7.8% |

HMM latent-state decoding accuracy: **0.813** (4 regimes; chance = 0.25).

### What holds, and what does not (honest)

- **Spoofing harms almost entirely *through trust*.** 96.8% of its DRAC effect is
  the trust-mediated indirect effect (`fig2`): the attack poisons compliance, and
  the crash risk lands later, on a legitimate warning the driver now ignores.
- **Suppression harms *directly*.** Only 7.8% runs through trust — the danger is
  the missing warning itself; trust stays near-intact (0.79 vs. baseline 0.96).
- **The rule baseline fails *selectively*.** The Static Compliance Rule is blind
  to spoofing (bias 4.39) yet essentially correct on suppression (bias 0.02),
  precisely because it can represent a missing warning but not a collapse of
  trust. This is the sharpest evidence that the trust channel is real and that a
  compliance-invariant model cannot see it.
- **The correlational baseline is biased, not useless.** Confounding by driver
  class inflates the naive estimate (5.26 vs. 4.58); AIPW removes it. We report
  the biased number rather than hiding it.
- **Mitigation only *partially* recovers safety.** Calibrated-uncertainty
  advisories lift mean trust to 0.48 (vs. 0.22 under naive binary spoofing) and
  cut the critical-TTC rate from 37.0% to 21.1% and mean DRAC from 6.82 to
  4.79 m/s² — a real but incomplete improvement (`fig3`), not a fix.
- **The decode is imperfect by construction.** Emission overlap caps HMM accuracy
  at ~0.81; real telemetry would be harder still.

## Figures (`figures/`, 300 DPI, svg/pdf/jpg)

1. `fig1_trust_hysteresis_loop` — latent compliance trajectory before / during /
   after the attack: fast collapse, slow partial recovery.
2. `fig2_causal_mediation_forest` — NDE vs. trust-mediated NIE, by attack and
   driver class (the mechanism contrast).
3. `fig3_drac_hazard_distributions` — peak-DRAC distributions across baseline /
   spoofed / suppressed / calibrated-mitigation.
4. `fig4_hmm_state_transitions` — decoded transition matrix + latent state
   occupancy by condition.

## Install & reproduce

```bash
pip install -r requirements.txt
pytest -q                                   # 7 tests: HMM, ATE/mediation recovery, figures
python experiments/run_behavioral_study.py  # 300 episodes -> results/*.csv (+ console tables)
python visualization/make_pub_figures.py    # figures/ at 300 DPI (svg/pdf/jpg)
```

Fully offline and deterministic (fixed seeds); no external data or API keys.

## Repository structure

```
sim/heterogeneous_driver_sim.py   heterogeneous fleet + trust dynamics + potential outcomes
models/trust_hmm.py               Gaussian HMM (Baum-Welch EM, Viterbi) latent-trust decoder
models/causal_mediation.py        IPW / AIPW ATE + NDE/NIE mediation (parametric g-formula)
experiments/run_behavioral_study.py  end-to-end study -> results/*.csv
visualization/make_pub_figures.py four 300-DPI publication figures
tests/test_behavioral_causal.py   HMM accuracy, causal recovery, mechanism contrast, export
```

## Limitations & threats to validity

- **Simulation, not field data.** Telemetry, the trust-update rule, and the
  attack mix are synthetic; absolute magnitudes are assumptions. The relative
  estimator comparison under a shared generator is the contribution.
- **Linear, no-interaction mediation.** NDE/NIE use a parametric g-formula that
  assumes no exposure–mediator interaction and correct nuisance models; real
  behavior may violate both.
- **Measured mediator.** We treat the decoded/true trust state as the mediator;
  in the field it must be recovered from kinematics (the HMM), which bounds
  real-world performance below this ceiling.

## License

MIT — see [LICENSE](LICENSE).
