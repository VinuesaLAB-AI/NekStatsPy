"""Shared plotting style (colour-blind-safe Okabe-Ito palette)."""

import matplotlib.pyplot as plt

# Okabe-Ito, fixed order, yellow dropped for white background legibility.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
             "#56B4E9", "#000000"]

RC = {
    "font.size": 13, "axes.labelsize": 15, "axes.linewidth": 1.1,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.8,
    "figure.dpi": 120,
}

# how each dataset layer is drawn
LINE_KW = dict(lw=2.4, alpha=0.85, zorder=2)
MARKER_KW = dict(ls="none", ms=6.5, mfc="none", mew=1.4, zorder=3)


def apply_style():
    plt.rcParams.update(RC)
