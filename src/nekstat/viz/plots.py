"""Plotting for velocity profiles, TKE budget and streamwise-developing quantities.

Every function takes one or more *layers*.  A layer is a dict
``{"result": Result, "label": str, "marker": bool}``; drawn as a solid line
(``marker=False``) or open markers (``marker=True``).  With a single layer you get
a normal plot; with two layers (e.g. a MATLAB reference and the Python port) you
get an overlay.  Convenience wrappers ``plot_single`` / ``plot_overlay`` build the
layers for you.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import OKABE_ITO, LINE_KW, MARKER_KW, apply_style

_BUDGET = [("Pkp", "Production"), ("Dkp", "Dissipation"),
           ("Tkp", "Turb. transport"), ("VDkp", "Visc. diffusion"),
           ("VPGkp", "Vel-press-grad"), ("Ckp", "Convection")]


def _as_layers(layers):
    """Normalise a Result / list of Results / list of layer-dicts to layer-dicts."""
    from ..result import Result
    if isinstance(layers, Result):
        layers = [layers]
    out = []
    for i, item in enumerate(layers):
        if isinstance(item, Result):
            out.append({"result": item, "label": item.name,
                        "marker": i > 0})
        else:
            d = dict(item)
            d.setdefault("marker", False)
            d.setdefault("label", d["result"].name)
            out.append(d)
    return out


def _draw(ax, layer, x, y, color, marker="o", markevery=12):
    if layer["marker"]:
        ax.plot(x, y, color=color, marker=marker, markevery=markevery, **MARKER_KW)
    else:
        ax.plot(x, y, color=color, **LINE_KW)


def _source_handles(layers):
    h = []
    for lyr in layers:
        if lyr["marker"]:
            h.append(Line2D([0], [0], color="0.3", ls="none", marker="o",
                            mfc="none", mew=1.4, ms=7, label=lyr["label"]))
        else:
            h.append(Line2D([0], [0], color="0.3", lw=2.4, label=lyr["label"]))
    return h


# ---------------------------------------------------------------------------
def plot_velocity(layers, stations=(0.50, 0.75), side="ss", savepath=None):
    """U and V, inner (U+,V+) and outer (U/Ue,V/Ue) scaling, at chord stations."""
    apply_style()
    layers = _as_layers(layers)
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for si, xc in enumerate(stations):
        c = OKABE_ITO[si]
        for lyr in layers:
            R = lyr["result"]
            p = R.station(side, xc)
            yp = R.yplus(side, p)
            ut = R.scal(side, "ut")[p]; Ue = R.scal(side, "Ue")[p]
            _draw(ax[0, 0], lyr, yp, R.prof(side, p, "U") / ut, c)
            _draw(ax[0, 1], lyr, yp, R.prof(side, p, "U") / Ue, c)
            _draw(ax[1, 0], lyr, yp, R.prof(side, p, "V") / ut, c)
            _draw(ax[1, 1], lyr, yp, R.prof(side, p, "V") / Ue, c)
    for a in ax.ravel():
        a.set_xscale("log")
    ax[0, 0].set(xlabel=r"$y_n^+$", ylabel=r"$U_t^+$", title="(a) inner")
    ax[0, 1].set(xlabel=r"$y_n^+$", ylabel=r"$U_t/U_e$", title="(b) outer")
    ax[1, 0].set(xlabel=r"$y_n^+$", ylabel=r"$V_n^+$", title="(c) inner")
    ax[1, 1].set(xlabel=r"$y_n^+$", ylabel=r"$V_n/U_e$", title="(d) outer")
    h = [Line2D([0], [0], color=OKABE_ITO[i], lw=2.4, label=rf"$x/c={s}$")
         for i, s in enumerate(stations)] + _source_handles(layers)
    ax[1, 0].legend(handles=h, loc="upper left", framealpha=0.9)
    fig.suptitle(f"Velocity profiles ({side.upper()})", fontsize=15, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return _finish(fig, savepath)


def plot_budget(layers, station=0.75, sides=("ss", "ps"), savepath=None):
    """Inner-scaled turbulent-kinetic-energy budget."""
    apply_style()
    layers = _as_layers(layers)
    fig, ax = plt.subplots(1, len(sides), figsize=(7.5 * len(sides), 6.2),
                           squeeze=False)
    ax = ax[0]
    names = {"ss": "Suction side", "ps": "Pressure side"}
    for k, side in enumerate(sides):
        a = ax[k]
        for lyr in layers:
            R = lyr["result"]
            p = R.station(side, station)
            yp = R.yplus(side, p)
            for j, (f, _) in enumerate(_BUDGET):
                _draw(a, lyr, yp, R.prof(side, p, f), OKABE_ITO[j], markevery=10)
        a.axhline(0, color="0.6", lw=0.9)
        a.set(xscale="log", xlabel=r"$y_n^+$",
              ylabel=r"gain / loss $\;\times\,\nu/u_\tau^4$",
              title=f"({'ab'[k]}) {names.get(side, side)},  $x/c={station}$",
              xlim=(0.4, 400))
    h = [Line2D([0], [0], color=OKABE_ITO[j], lw=2.4, label=lab)
         for j, (_, lab) in enumerate(_BUDGET)] + _source_handles(layers)
    ax[0].legend(handles=h, loc="lower left", ncol=2, framealpha=0.9, fontsize=10)
    fig.suptitle("Turbulent-kinetic-energy budget (inner-scaled)", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, savepath)


def plot_streamwise(layers, savepath=None):
    """Cf (both sides), beta, Re_theta, Re_tau, H12 vs x/c."""
    apply_style()
    layers = _as_layers(layers)
    quants = [("Cf", r"$c_f$", False, True), ("beta", r"$\beta$", False, False),
              ("Reth", r"$Re_\theta$", True, False),
              ("Ret", r"$Re_\tau$", True, False),
              ("H12", r"$H_{12}$", False, False)]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ax = ax.ravel()
    for i, (f, lab, ylog, bothsides) in enumerate(quants):
        a = ax[i]
        for side, color, ls in [("ss", OKABE_ITO[0], "-"),
                                ("ps", OKABE_ITO[1], "--")]:
            if side == "ps" and not bothsides:
                continue
            for lyr in layers:
                R = lyr["result"]
                x = R.scal(side, "xa"); y = R.scal(side, f)
                if lyr["marker"]:
                    a.plot(x, y, color=color, marker="o", markevery=2, **MARKER_KW)
                else:
                    a.plot(x, y, color=color, ls=ls, **LINE_KW)
        if ylog:
            a.set_yscale("log")
        a.set(xlabel=r"$x/c$", ylabel=lab, title=f"({'abcdef'[i]}) {lab}")
    handles = [Line2D([0], [0], color=OKABE_ITO[0], lw=2.4, label="suction side"),
               Line2D([0], [0], color=OKABE_ITO[1], lw=2.4, ls="--",
                      label="pressure side")] + _source_handles(layers)
    ax[-1].axis("off")
    ax[-1].legend(handles=handles, loc="center", framealpha=0.9)
    fig.suptitle("Streamwise-developing quantities", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _finish(fig, savepath)


def _finish(fig, savepath):
    if savepath:
        os.makedirs(os.path.dirname(os.path.abspath(savepath)), exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
        return savepath
    return fig


# ---- convenience wrappers --------------------------------------------------
def plot_single(result, figdir, prefix="stat"):
    """Produce the three figures for a single dataset (solid lines)."""
    layers = [{"result": result, "label": result.name, "marker": False}]
    return [
        plot_velocity(layers, savepath=os.path.join(figdir, f"{prefix}_velocity.png")),
        plot_budget(layers, savepath=os.path.join(figdir, f"{prefix}_budget.png")),
        plot_streamwise(layers, savepath=os.path.join(figdir, f"{prefix}_streamwise.png")),
    ]


def plot_overlay(ref, mine, figdir, ref_label="MATLAB (ref)",
                 mine_label="Python (port)"):
    """Overlay two datasets (reference as lines, `mine` as markers)."""
    layers = [{"result": ref, "label": ref_label, "marker": False},
              {"result": mine, "label": mine_label, "marker": True}]
    return [
        plot_velocity(layers, savepath=os.path.join(figdir, "01_velocity_profiles.png")),
        plot_budget(layers, savepath=os.path.join(figdir, "02_tke_budget.png")),
        plot_streamwise(layers, savepath=os.path.join(figdir, "03_streamwise.png")),
    ]
