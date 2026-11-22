"""Causal effect and mediation estimators for the advisory-spoofing study.

Implements, from scratch (statsmodels/sklearn for the nuisance fits only):

* `naive_ate`  -- unadjusted difference in means (biased under confounding);
* `ipw_ate`    -- inverse-propensity-weighted ATE;
* `aipw_ate`   -- augmented IPW (doubly-robust) ATE with bootstrap CI;
* `mediation_effects` -- regression-based decomposition of the total effect into
  a Natural Direct Effect (NDE) and a trust-mediated Natural Indirect Effect
  (NIE) via the parametric g-formula (VanderWeele), with bootstrap CIs.

`make_synthetic_causal` emits a linear structural model with *known* ATE / NDE /
NIE and built-in confounding, used to validate recovery in the test-suite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression


# ---------------------------------------------------------------------------
# synthetic ground-truth structural model (for validation)
# ---------------------------------------------------------------------------
def make_synthetic_causal(n: int = 4000, seed: int = 0):
    """Linear SCM with confounding.  Returns (df, truth-dict).

        X ~ N(0,1)                      (confounder, e.g. aggressiveness)
        T ~ Bern(sigmoid(0.9 X))        (treatment assignment depends on X)
        M = 0.5 + 1.2 T + 0.8 X + e_m   (trust mediator)
        Y = 1.0 + 0.7 T + 1.5 M + 0.6 X + e_y

        NDE = 0.7, NIE = 1.5*1.2 = 1.8, ATE = 2.5
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    T = (rng.random(n) < 1 / (1 + np.exp(-0.9 * X))).astype(int)
    M = 0.5 + 1.2 * T + 0.8 * X + rng.normal(0, 0.5, n)
    Y = 1.0 + 0.7 * T + 1.5 * M + 0.6 * X + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"X": X, "T": T, "M": M, "Y": Y})
    truth = {"ATE": 2.5, "NDE": 0.7, "NIE": 1.8}
    return df, truth


# ---------------------------------------------------------------------------
# ATE estimators
# ---------------------------------------------------------------------------
def _design(df, covariates):
    return df[covariates].to_numpy(float) if covariates else np.zeros((len(df), 0))


def naive_ate(df, treat="T", outcome="Y"):
    g = df.groupby(treat)[outcome].mean()
    return float(g.get(1, np.nan) - g.get(0, np.nan))


def _propensity(df, treat, covariates):
    X = _design(df, covariates)
    if X.shape[1] == 0:
        p = np.full(len(df), df[treat].mean())
    else:
        lr = LogisticRegression(max_iter=1000).fit(X, df[treat].to_numpy())
        p = lr.predict_proba(X)[:, 1]
    return np.clip(p, 0.02, 0.98)


def ipw_ate(df, treat="T", outcome="Y", covariates=("X",)):
    covariates = list(covariates)
    e = _propensity(df, treat, covariates)
    t, y = df[treat].to_numpy(), df[outcome].to_numpy(float)
    w1, w0 = t / e, (1 - t) / (1 - e)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def _aipw_point(df, treat, outcome, covariates):
    covariates = list(covariates)
    t = df[treat].to_numpy()
    y = df[outcome].to_numpy(float)
    e = _propensity(df, treat, covariates)
    X = _design(df, covariates)
    # outcome regressions per arm
    if X.shape[1] == 0:
        mu1 = np.full(len(df), y[t == 1].mean())
        mu0 = np.full(len(df), y[t == 0].mean())
    else:
        m1 = LinearRegression().fit(X[t == 1], y[t == 1])
        m0 = LinearRegression().fit(X[t == 0], y[t == 0])
        mu1, mu0 = m1.predict(X), m0.predict(X)
    psi = (mu1 - mu0
           + t * (y - mu1) / e
           - (1 - t) * (y - mu0) / (1 - e))
    return float(psi.mean())


def aipw_ate(df, treat="T", outcome="Y", covariates=("X",), n_boot=400, seed=0):
    est = _aipw_point(df, treat, outcome, covariates)
    rng = np.random.default_rng(seed)
    boots = []
    idx = np.arange(len(df))
    for _ in range(n_boot):
        b = df.iloc[rng.choice(idx, len(df), replace=True)]
        try:
            boots.append(_aipw_point(b, treat, outcome, covariates))
        except Exception:                      # pragma: no cover
            continue
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"ate": est, "ci_low": float(lo), "ci_high": float(hi),
            "se": float(np.std(boots))}


# ---------------------------------------------------------------------------
# causal mediation (parametric g-formula; linear, no T-M interaction)
# ---------------------------------------------------------------------------
def _mediation_point(df, treat, mediator, outcome, covariates):
    cov = list(covariates)
    t = df[treat].to_numpy(float)
    m = df[mediator].to_numpy(float)
    C = _design(df, cov)

    # mediator model:  M ~ T + C   -> alpha_T = effect of T on M
    Xm = np.column_stack([t, C]) if C.shape[1] else t[:, None]
    am = LinearRegression().fit(Xm, m)
    alpha_T = am.coef_[0]

    # outcome model:   Y ~ T + M + C  -> beta_T (direct), beta_M
    Xy = np.column_stack([t, m, C]) if C.shape[1] else np.column_stack([t, m])
    ay = LinearRegression().fit(Xy, df[outcome].to_numpy(float))
    beta_T, beta_M = ay.coef_[0], ay.coef_[1]

    nde = float(beta_T)
    nie = float(beta_M * alpha_T)
    total = nde + nie
    return {"NDE": nde, "NIE": nie, "ATE": total,
            "prop_mediated": float(nie / total) if total != 0 else np.nan}


def mediation_effects(df, treat="T", mediator="M", outcome="Y",
                      covariates=("X",), n_boot=400, seed=0):
    point = _mediation_point(df, treat, mediator, outcome, covariates)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    keys = ["NDE", "NIE", "ATE", "prop_mediated"]
    boots = {k: [] for k in keys}
    for _ in range(n_boot):
        b = df.iloc[rng.choice(idx, len(df), replace=True)]
        try:
            p = _mediation_point(b, treat, mediator, outcome, covariates)
        except Exception:                      # pragma: no cover
            continue
        for k in keys:
            boots[k].append(p[k])
    out = dict(point)
    for k in keys:
        lo, hi = np.percentile(boots[k], [2.5, 97.5])
        out[f"{k}_ci"] = (float(lo), float(hi))
    return out


if __name__ == "__main__":
    df, truth = make_synthetic_causal(seed=1)
    print("truth:", truth)
    print("naive ATE :", round(naive_ate(df), 3), "(confounded, biased high)")
    print("IPW   ATE :", round(ipw_ate(df), 3))
    a = aipw_ate(df, n_boot=200)
    print("AIPW  ATE :", round(a["ate"], 3),
          f"[{a['ci_low']:.2f}, {a['ci_high']:.2f}]")
    med = mediation_effects(df, n_boot=200)
    print("mediation : NDE=%.3f NIE=%.3f prop_med=%.2f"
          % (med["NDE"], med["NIE"], med["prop_mediated"]))
