"""Publication figures (dark theme, Times New Roman, 300 DPI, svg/pdf/jpg).

  fig1  trust / compliance hysteresis loop (asymmetric collapse & recovery)
  fig2  causal-mediation forest: NDE vs trust-mediated NIE, by attack & class
  fig3  peak-DRAC hazard distributions across advisory conditions
  fig4  HMM transition heatmap + latent state-occupancy stack
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

FIG_BG, AX_BG, FG, GRID = "#0f0f1a", "#1a1a2e", "#d6d6e0", "#33334d"
C_OURS, C_STD, C_RULE, C_EM, C_PUR = "#ffd166", "#ff6b6b", "#4cc9f0", "#06d6a0", "#b983ff"
COND_COLOR = {"baseline": C_EM, "spoofed": C_STD, "suppressed": C_PUR,
              "calibrated": C_OURS}
CLASS_COLOR = {"aggressive": C_STD, "cautious": C_EM, "cav_assisted": C_RULE}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "figure.facecolor": FIG_BG, "axes.facecolor": AX_BG, "savefig.facecolor": FIG_BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": FG, "ytick.color": FG, "grid.color": GRID, "grid.alpha": 0.5,
    "axes.grid": True, "axes.axisbelow": True, "font.size": 11,
})


def _save(fig, name):
    for ext in ("svg", "pdf", "jpg"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return ax


def fig1_trust_hysteresis_loop():
    df = pd.read_csv(os.path.join(RESULTS, "trust_trajectories.csv"))
    n_pre = 5                                    # attack window length
    fig, ax = plt.subplots(figsize=(8.6, 5))
    for cls, g in df.groupby("driver_class"):
        g = g.sort_values("encounter")
        ax.plot(g.encounter, g.trust, "-o", ms=4, lw=2.2,
                color=CLASS_COLOR[cls], label=cls)
        ax.fill_between(g.encounter, g.trust - g.trust_sd, g.trust + g.trust_sd,
                        color=CLASS_COLOR[cls], alpha=0.12)
    emax = int(df.encounter.max())
    ax.axvspan(-0.3, n_pre - 0.5, color=C_STD, alpha=0.08)
    ax.axvspan(n_pre - 0.5, emax + 0.3, color=C_EM, alpha=0.06)
    ax.text(2.0, 0.93, "spoofing attack\n(false advisories)", color=C_STD,
            ha="center", fontsize=9)
    ax.text((n_pre + emax) / 2, 0.93, "recovery window\n(legitimate advisories)",
            color=C_EM, ha="center", fontsize=9)
    ax.axhline(0.5, ls="--", lw=1, color=FG, alpha=0.4)
    ax.annotate("sharp trust\ncollapse", xy=(2.5, 0.45), xytext=(3.4, 0.6),
                color=FG, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=FG, alpha=0.7))
    ax.annotate("slow, partial\nrecovery (hysteresis)", xy=(7, 0.33),
                xytext=(5.2, 0.12), color=FG, fontsize=9,
                arrowprops=dict(arrowstyle="->", color=FG, alpha=0.7))
    _style(ax)
    ax.set_xlabel("advisory encounter (time ->)")
    ax.set_ylabel("latent trust / compliance propensity")
    ax.set_ylim(0, 1.0)
    ax.set_title("Compliance hysteresis under spoofed corridor advisories\n"
                 "trust collapses fast but recovers slowly after cyber recovery",
                 color=FG)
    ax.legend(frameon=False, fontsize=9, title="driver class",
              title_fontsize=9)
    _save(fig, "fig1_trust_hysteresis_loop")


def fig2_causal_mediation_forest():
    df = pd.read_csv(os.path.join(RESULTS, "causal_mediation.csv"))
    rows = []   # (label, value, lo, hi, color, kind)
    # spoofed overall: NDE vs NIE
    sp = df[(df.contrast == "spoofed_vs_baseline") & (df.driver_class == "ALL")]
    su = df[(df.contrast == "suppressed_vs_baseline") & (df.driver_class == "ALL")]
    if len(sp):
        r = sp.iloc[0]
        rows.append(("Spoofed - direct (NDE)", r.NDE, r.NDE_lo, r.NDE_hi, C_RULE))
        rows.append(("Spoofed - trust-mediated (NIE)", r.NIE, r.NIE_lo, r.NIE_hi,
                     C_OURS))
    # spoofed per class NIE
    for cls in ["aggressive", "cautious", "cav_assisted"]:
        g = df[(df.contrast == "spoofed_vs_baseline") & (df.driver_class == cls)]
        if len(g):
            r = g.iloc[0]
            rows.append((f"   {cls}: NIE", r.NIE, r.NIE_lo, r.NIE_hi, C_OURS))
    if len(su):
        r = su.iloc[0]
        rows.append(("Suppressed - direct (NDE)", r.NDE, r.NDE_lo, r.NDE_hi,
                     C_RULE))
        rows.append(("Suppressed - trust-mediated (NIE)", r.NIE, r.NIE_lo,
                     r.NIE_hi, C_OURS))

    labels = [r[0] for r in rows]
    y = np.arange(len(rows))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for yi, (_, val, lo, hi, color) in zip(y, rows):
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.4, alpha=0.9)
        ax.plot(val, yi, "o", color=color, ms=8)
        ax.text(hi + 0.08, yi, f"{val:.2f}", va="center", color=FG, fontsize=8.5)
    ax.axvline(0, ls="--", lw=1, color=FG, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    _style(ax)
    ax.set_xlabel("effect on peak DRAC (m/s$^2$)  [95% bootstrap CI]")
    ax.set_title("Causal mediation: spoofing harms via trust, suppression "
                 "harms directly\n(amber = trust-mediated indirect effect; "
                 "blue = direct effect)", color=FG)
    _save(fig, "fig2_causal_mediation_forest")


def fig3_drac_hazard_distributions():
    df = pd.read_csv(os.path.join(RESULTS, "safety_surrogates.csv"))
    order = ["baseline", "spoofed", "suppressed", "calibrated"]
    data = [df[df.condition == c]["peak_drac"].to_numpy() for c in order]
    fig, ax = plt.subplots(figsize=(8.6, 5))
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.8)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COND_COLOR[order[i]])
        pc.set_edgecolor(COND_COLOR[order[i]])
        pc.set_alpha(0.45)
    parts["cmeans"].set_color(FG)
    ax.axhline(3.4, ls="--", lw=1.2, color=FG, alpha=0.6)
    ax.text(0.6, 3.6, "comfortable braking (3.4 m/s$^2$)", color=FG, fontsize=8.5)
    _style(ax)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(["Baseline", "Spoofed", "Suppressed",
                        "Calibrated\n(mitigation)"])
    ax.set_ylabel("peak DRAC at hazard (m/s$^2$)")
    ax.set_title("Surrogate crash-risk distributions by advisory condition\n"
                 "calibrated-uncertainty advisories recover much of the loss "
                 "under attack", color=FG)
    _save(fig, "fig3_drac_hazard_distributions")


def fig4_hmm_state_transitions():
    A = pd.read_csv(os.path.join(RESULTS, "hmm_transitions.csv"), index_col=0)
    occ = pd.read_csv(os.path.join(RESULTS, "hmm_occupancy.csv"))
    regimes = list(A.columns)
    short = [r.replace("_", "\n") for r in regimes]
    fig, (axh, axb) = plt.subplots(1, 2, figsize=(12.5, 5),
                                   gridspec_kw={"width_ratios": [1, 1.15]})
    im = axh.imshow(A.to_numpy(), cmap="magma", vmin=0, vmax=1, aspect="auto")
    axh.set_xticks(range(len(regimes))); axh.set_xticklabels(short, fontsize=8)
    axh.set_yticks(range(len(regimes))); axh.set_yticklabels(short, fontsize=8)
    axh.set_xlabel("to state"); axh.set_ylabel("from state")
    axh.set_title("Decoded latent transition matrix", color=FG)
    axh.grid(False)
    for i in range(len(regimes)):
        for j in range(len(regimes)):
            v = A.to_numpy()[i, j]
            axh.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if v < 0.6 else "black")
    fig.colorbar(im, ax=axh, fraction=0.046, pad=0.04)

    conds = occ["condition"].tolist()
    bottom = np.zeros(len(conds))
    pal = [C_EM, C_RULE, C_PUR, C_STD]
    for k, reg in enumerate(regimes):
        vals = occ[reg].to_numpy()
        axb.bar(conds, vals, bottom=bottom, color=pal[k], label=reg.replace("_", " "),
                edgecolor=FIG_BG, lw=0.6)
        bottom += vals
    _style(axb)
    axb.set_ylabel("latent state occupancy")
    axb.set_title("State occupancy by advisory condition", color=FG)
    axb.legend(frameon=False, fontsize=8, ncol=1, loc="center left",
               bbox_to_anchor=(1.0, 0.5))
    axb.tick_params(axis="x", rotation=20)
    fig.suptitle("Latent trust-state dynamics recovered by the Gaussian HMM",
                 color=FG, fontsize=13)
    fig.tight_layout()
    _save(fig, "fig4_hmm_state_transitions")


def main():
    print("Generating publication figures (300 DPI) ...")
    fig1_trust_hysteresis_loop()
    fig2_causal_mediation_forest()
    fig3_drac_hazard_distributions()
    fig4_hmm_state_transitions()
    print("Figures written to", FIGDIR)


if __name__ == "__main__":
    main()
