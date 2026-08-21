"""C4 -- signed deviation from the reference on a symlog axis.

One linear axis cannot hold both the low-order transient and the endgame, which
is why the raw convergence figure needs a ``_tight`` twin.  A symlog axis holds
both: outside the linear band it is logarithmic, so you read the transient and
the SIGN of the approach; inside the band it turns linear, so the endgame is
legible in the same panel.

The band half-width is ``GRCWA_SYMLOG_LINTHRESH``, default 1e-6.  It sets how
far down the log region reaches before the axis goes linear, so a smaller value
buys resolution in the endgame at the cost of collapsing everything below it
into one flat zone.  1e-6 keeps the 1e-4 tolerance itself on the log side, where
its decade is still readable; setting it to 1e-4 puts the tolerance at the band
edge instead, which hides how a curve approaches it.  The tick decades follow
automatically.

Two variants of the same figure, chosen by the x axis: total retained orders,
and wall time of the solve.  Moose is overlaid in black on both, from
``moose_reference.json`` and the per-point ``t_solve_s`` in
``moose_timing.json``.

Where a case is judged against Moose, its own curve is a SELF-convergence
curve -- it reaches the reference by construction -- and says how fast Moose
settles, not how accurate it is.  On D1 and D2, which are judged against
Ikarus's normal-vector value, it is a genuine third-code accuracy curve.
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
def _linthresh():
    """Half-width of the symlog linear band, from GRCWA_SYMLOG_LINTHRESH."""
    raw = os.environ.get("GRCWA_SYMLOG_LINTHRESH")
    if not raw:
        return 1e-6
    try:
        value = float(raw)
    except ValueError:
        raise SystemExit("GRCWA_SYMLOG_LINTHRESH is not a number: %r" % raw)
    if not 0 < value < 1:
        raise SystemExit("GRCWA_SYMLOG_LINTHRESH must be in (0, 1): %r" % raw)
    return value


LT = _linthresh()      # symlog threshold: where the axis stops being logarithmic


def _decades(lt):
    """Decade tick magnitudes from 0.1 down to the last one above the band."""
    lowest = int(np.ceil(np.log10(lt)))
    return [10.0 ** e for e in range(-1, lowest, -1)]


def _tick_label(value):
    """0.1 and 0.01 read better spelled out; below that use the exponent."""
    if value >= 0.01:
        return ("%g" % value).rstrip("0").rstrip(".")
    return "1e%d" % round(np.log10(value))
BAND = "#e6f4ec"


def _ref_note(case, mention_moose=True):
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


def panel(ax, case, xaxis):
    Rref, rtype, prov = D.ref_of(case)
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
    else:
        mx, mR = D.moose_timed(case)
    if mx is not None and len(mx):
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
    mags = [1.0] + _decades(LT)
    ax.set_yticks([-v for v in mags] + [0] + list(reversed(mags)))
    ax.set_yticklabels(
        ["−" + _tick_label(v) for v in mags] + ["0"] +
        ["+" + _tick_label(v) for v in reversed(mags)], fontsize=7.6)
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
    fig, axes = plt.subplots(nrow, ncol, figsize=(17.5, 3.55 * nrow))
    fig.subplots_adjust(left=.055, right=.985, top=.878, bottom=.085,
                        hspace=.60, wspace=.20)
    for ax in axes.flat:
        ax.axis("off")
    any_moose = False
    for i, case in enumerate(cases):
        ax = axes.flat[i]
        ax.axis("on")
        _prov, hm = panel(ax, case, xaxis)
        any_moose |= hm
        _header(ax, case)
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
                              label="Moose (independent code)"))
    handles.append(Line2D([], [], color=BAND, lw=9,
                          label="±%s band (linear zone)" % _tick_label(LT)))
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
               "— Moose from its own per-point t_solve, so the independent code is on the "
               "cost axis too")
    fig.suptitle(head, fontsize=17, fontweight="bold", color=INK, y=.972)
    fig.text(.5, .922, sub, ha="center", fontsize=10.3, color=MUTED, linespacing=1.5)
    name = "conv_deviation_%s.png" % xaxis
    fig.savefig(os.path.join(OUT, name), dpi=112)
    plt.close(fig)
    print("wrote", os.path.join(OUT, name))


if __name__ == "__main__":
    figure("orders")
    figure("time")
