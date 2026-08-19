"""The cost staircase: the best |R - R_ref| a wall-time budget can buy.

    python benchmark/plot_conv_cost.py      -> conv_cost_staircase.png

Faint markers are the raw measurements; the thick step is the running best, i.e.
the honest answer to "with this much time, how close can I get?".  It is
monotone by construction, so a rule that oscillates over nG becomes a flat tread
rather than a scribble -- which is what makes this readable where a plain
error-vs-time scatter is not.  A flat tread means spending more buys nothing.

Colour encodes the factorization rule, dash and marker the codebase
(see viz_conv.py).  ``GRCWA_CONV_JSON`` picks the run, ``GRCWA_PLOT_OUTPUT_DIR``
the output directory.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtr
from matplotlib.lines import Line2D

import viz_conv as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("GRCWA_PLOT_OUTPUT_DIR", HERE)
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, MUTED = "#1b2733", "#5b6b7b"
TOL = [(1e-2, "1%"), (1e-3, "1e-3"), (1e-4, "1e-4")]
FLOOR = 1e-12


def _ref_note(case, mention_moose=False):
    """Short lines naming the reference a panel is measured against."""
    R, kind, prov = D.ref_of(case)
    src = {"external_moose": "Moose", "analytic_exact": "analytic (Airy)"}.get(
        kind, (D.CASES[case].get("ref") or {}).get("from") or kind)
    head = "ref = %s %.6g" % (src, R)
    if prov:
        head += "  ·  provisional"
    lines = [head]
    if mention_moose and kind == "external_moose":
        lines.append("black curve is Moose itself → self-convergence")
    return lines


def _header(ax, case, fs=10.5):
    lines = _ref_note(case)
    ax.set_title(case, fontsize=fs, fontweight="bold", color=INK, loc="left",
                 pad=7 + 10.0 * len(lines))
    ax.text(0, 1.015, "\n".join(lines), transform=ax.transAxes, fontsize=7.6,
            color=MUTED, va="bottom", ha="left", linespacing=1.4)


def panel(ax, case):
    xmin, xmax = np.inf, 0.0
    for col in D.COLUMNS:
        nG, R, err, sgn, traw, test = D.series(case, col)
        ok = np.isfinite(test) & (test > 0)
        x, e = test[ok], np.maximum(err[ok], FLOOR)
        if not len(x):
            continue
        st = D.style(col)
        ax.plot(x, e, ls="none", marker=st["marker"], ms=3.0, mfc="none",
                mec=st["color"], mew=.8, alpha=.45, zorder=3)
        xs, best = D.pareto(x, e)
        ax.step(xs, best, where="post", color=st["color"], ls=st["linestyle"],
                lw=2.2, zorder=5, solid_capstyle="round")
        xmin, xmax = min(xmin, xs.min()), max(xmax, xs.max())
        hit = np.nonzero(best <= 1e-4)[0]
        if len(hit):
            ax.plot([xs[hit[0]]], [1e-4], marker="v", ms=7, color=st["color"],
                    mec="white", mew=.8, zorder=7)
    if not np.isfinite(xmin):
        return
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(xmin * .7, xmax * 1.5)
    lo, hi = ax.get_ylim()
    ax.set_ylim(max(lo, 1e-9), min(hi, 3))
    tb = mtr.blended_transform_factory(ax.transAxes, ax.transData)
    ylo, yhi = ax.get_ylim()
    for v, lab in TOL:
        if not (ylo < v < yhi):
            continue
        ax.axhline(v, color="#b9c7d4", lw=.8, ls=(0, (4, 3)), zorder=1)
        ax.text(.995, v, lab, va="center", ha="right", fontsize=7.2, color=MUTED,
                transform=tb, zorder=8,
                bbox=dict(fc="white", ec="none", alpha=.75, pad=.6))
    ax.axhspan(ylo, 1e-4, color="#eaf5ee", zorder=0)
    ax.grid(True, which="major", color="#eef2f6", lw=.7, zorder=0)
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3ced8")


def figure(fname="conv_cost_staircase.png", ncol=3, dpi=112):
    cases = D.ORDER
    nrow = -(-len(cases) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(17.5, 3.6 * nrow))
    fig.subplots_adjust(left=.055, right=.955, top=.878, bottom=.085,
                        hspace=.58, wspace=.30)
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for i, case in enumerate(cases):
        ax = np.atleast_1d(axes).flat[i]
        ax.axis("on")
        panel(ax, case)
        _header(ax, case)
        if i >= len(cases) - ncol:
            ax.set_xlabel("wall time of the solve [ms]", fontsize=8.6, color=MUTED)
        if i % ncol == 0:
            ax.set_ylabel("|R − R_ref|", fontsize=8.6, color=MUTED)
    handles = [Line2D([], [], color=D.RULE_COLOR[D.rule_of(c)],
                      ls=D.CODE_DASH[D.code_of(c)], marker=D.CODE_MARK[D.code_of(c)],
                      ms=5, lw=2.2, label=c) for c in D.COLUMNS]
    handles.append(Line2D([], [], color="#666", marker="v", ls="none", ms=7,
                          label="running best first reaches 1e-4"))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               fontsize=9.6, bbox_to_anchor=(.5, .012))
    fig.suptitle("The cost staircase: the best |ΔR| a wall-time budget can buy",
                 fontsize=17, fontweight="bold", color=INK, y=.966)
    fig.text(.5, .925, "faint markers are the raw measurements; the thick step is the "
             "running best — a flat tread means spending more time buys nothing\n"
             "colour = factorization rule (the physics), dash + marker = codebase",
             ha="center", fontsize=10.3, color=MUTED, linespacing=1.5)
    path = os.path.join(OUT, fname)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    figure()
