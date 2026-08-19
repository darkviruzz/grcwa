"""C4 -- signed deviation from the reference on a symlog axis.

One linear axis cannot hold both the low-order transient and the 1e-4 endgame,
which is why the raw convergence figure needs a ``_tight`` twin.  A symlog axis
holds both: outside +/-1e-4 it is logarithmic, so you read the transient and the
SIGN of the approach; inside the band it turns linear, so the endgame is legible
in the same panel.

Two variants of the same figure, chosen by the x axis:
  * ``orders`` -- total retained orders.  Moose is overlaid in black wherever it
    has a sweep for the case, which makes this the cross-code comparison.
  * ``time``   -- wall time of the solve.  Moose carries no timings, so the
    external reference cannot appear here; the codes still can.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import viz_conv as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("GRCWA_PLOT_OUTPUT_DIR", HERE)
os.makedirs(OUT, exist_ok=True)


plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, MUTED = "#1b2733", "#5b6b7b"
LT = 1e-4              # symlog threshold: the tolerance the sweep is judged on
BAND = "#e6f4ec"


def panel(ax, case, xaxis):
    Rref, _rt, prov = D.ref_of(case)
    has_moose = False
    for col in D.COLUMNS:
        nG, R, err, sgn, traw, test = D.series(case, col)
        x = nG if xaxis == "orders" else test
        ok = np.isfinite(x) & (x > 0)
        st = D.style(col)
        ax.plot(x[ok], sgn[ok], color=st["color"], ls=st["linestyle"],
                marker=st["marker"], ms=3.4, lw=1.7, mec="none", zorder=5)
    if xaxis == "orders":
        mx, mR, _ = D.moose_series(case)
        if mx is not None:
            has_moose = True
            ax.plot(mx, mR - Rref, color=D.MOOSE_C, ls="-", marker="D", ms=5.0,
                    lw=2.4, mfc="white", mew=1.4, zorder=7)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=LT, linscale=.85)
    ax.axhspan(-LT, LT, color=BAND, zorder=1)
    ax.axhline(0, color="#9aa7b3", lw=.9, zorder=2)
    for v in (LT, -LT):
        ax.axhline(v, color="#7fbf9a", lw=.9, ls=(0, (4, 3)), zorder=3)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, -1e-1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["−1", "−0.1", "−0.01", "−1e-3", "0", "+1e-3", "+0.01",
                        "+0.1", "+1"], fontsize=7.6)
    ax.grid(axis="x", color="#eef2f6", lw=.7, zorder=0)
    ax.tick_params(axis="x", labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3ced8")
    return prov, has_moose


def figure(xaxis):
    cases = D.ORDER
    ncol = 3
    nrow = -(-len(cases) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(17.5, 3.35 * nrow))
    fig.subplots_adjust(left=.055, right=.985, top=.885, bottom=.085,
                        hspace=.52, wspace=.20)
    for ax in axes.flat:
        ax.axis("off")
    any_moose = False
    for i, case in enumerate(cases):
        ax = axes.flat[i]
        ax.axis("on")
        prov, hm = panel(ax, case, xaxis)
        any_moose |= hm
        tag = "   (provisional ref)" if prov else ""
        ax.set_title(case + tag, fontsize=10.5, fontweight="bold", color=INK,
                     loc="left", pad=6)
        if i % ncol == 0:
            ax.set_ylabel("R − R_ref   (symlog)", fontsize=8.8, color=MUTED)
        if i >= len(cases) - ncol:
            ax.set_xlabel("total retained orders" if xaxis == "orders"
                          else "wall time of the solve  [ms]",
                          fontsize=8.8, color=MUTED)

    handles = [Line2D([], [], color=D.RULE_COLOR[D.rule_of(c)],
                      ls=D.CODE_DASH[D.code_of(c)], marker=D.CODE_MARK[D.code_of(c)],
                      ms=5, lw=1.8, label=c) for c in D.COLUMNS]
    if any_moose:
        handles.append(Line2D([], [], color=D.MOOSE_C, marker="D", ms=6, lw=2.4,
                              mfc="white", mew=1.4,
                              label="Moose (independent reference code)"))
    handles.append(Line2D([], [], color=BAND, lw=9, label="±1e-4 band (linear zone)"))
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False,
               fontsize=9.6, bbox_to_anchor=(.5, .010))
    if xaxis == "orders":
        head = "Deviation from the reference vs retained orders (symlog)"
        sub = ("above the band you read the transient and the sign of the approach; inside "
               "the band the axis goes linear, so the endgame is legible in the same panel\n"
               "— one figure instead of the raw + tight pair, with Moose overlaid wherever "
               "the independent code has a sweep")
    else:
        head = "Deviation from the reference vs wall time (symlog)"
        sub = ("the same axis on the cost variable: how much of the deviation is still there "
               "per millisecond spent\n"
               "— Moose is absent here on purpose, the external reference carries no timings")
    fig.suptitle(head, fontsize=17, fontweight="bold", color=INK, y=.972)
    fig.text(.5, .922, sub, ha="center", fontsize=10.3, color=MUTED, linespacing=1.5)
    name = "conv_deviation_%s.png" % xaxis
    fig.savefig(os.path.join(OUT, name), dpi=112)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


if __name__ == "__main__":
    figure("orders")
    figure("time")
