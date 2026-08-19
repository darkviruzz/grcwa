"""C3 -- the scoreboard: what it costs each rule to reach 1e-4, in orders and in ms."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import viz_conv as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)



plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, MUTED = "#1b2733", "#5b6b7b"
TOL = 1e-4
CASES = D.ORDER[::-1]
OFF = np.linspace(.30, -.30, len(D.COLUMNS))

fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.4), sharey=True)
fig.subplots_adjust(left=.165, right=.985, top=.845, bottom=.135, wspace=.055)

for k, (ax, which, xlab) in enumerate(
        [(axes[0], "orders", "total retained orders needed"),
         (axes[1], "time", "wall time of that solve  [ms]")]):
    xs_all = []
    for i, case in enumerate(CASES):
        for j, col in enumerate(D.COLUMNS):
            a = D.arrival(case, col, TOL)
            if a is None:
                continue
            v = a[0] if which == "orders" else a[1]
            if not np.isfinite(v) or v <= 0:
                continue
            xs_all.append(v)
    lo = max(min(xs_all) * .55, 1e-1)
    hi = max(xs_all) * 1.9
    never_x = hi * 2.4
    for i, case in enumerate(CASES):
        ax.axhspan(i - .5, i + .5, color="#f7f9fb" if i % 2 else "white", zorder=0)
        got = []
        for j, col in enumerate(D.COLUMNS):
            a = D.arrival(case, col, TOL)
            st = D.style(col)
            y = i + OFF[j]
            if a is None:
                ax.plot([never_x], [y], marker="x", ms=7, mew=2.0,
                        color=st["color"], zorder=5, clip_on=False)
                continue
            v = a[0] if which == "orders" else a[1]
            got.append(v)
            prov = a[2] == "provisional"
            ax.plot([lo, v], [y, y], color=st["color"], lw=1.0, alpha=.30, zorder=3)
            ax.plot([v], [y], marker=st["marker"], ms=8.5,
                    mfc="white" if prov else st["color"], mec=st["color"],
                    mew=2.0 if prov else 1.0, zorder=6)
        if got:
            ax.plot([min(got)], [i + .48], marker="|", ms=0)
    ax.set_xscale("log")
    ax.set_xlim(lo, never_x * 1.45)
    ax.set_xticks([10.0 ** k for k in range(-1, 6)])
    ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs=np.arange(2, 10)))
    ax.set_xticks([tk for tk in ax.get_xticks() if lo <= tk <= hi])
    ax.set_ylim(-.6, len(CASES) - .4)
    ax.axvline(never_x, color="#e6ebf0", lw=18, zorder=1)
    ax.text(never_x, len(CASES) - .35, "never\nreaches 1e-4", ha="center", va="bottom",
            fontsize=8.4, color="#93a1ae", fontweight="bold", linespacing=1.3)
    ax.set_xlabel(xlab, fontsize=10.5, color=INK)
    ax.grid(axis="x", color="#e9eef3", lw=.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3ced8")
    ax.tick_params(labelsize=9)

axes[0].set_yticks(range(len(CASES)))
axes[0].set_yticklabels(CASES, fontsize=10, fontweight="bold", color=INK)
for i, case in enumerate(CASES):
    Rr, rt, prov = D.ref_of(case)
    axes[0].text(-.012, i - .30, ("provisional ref" if prov else "settled ref"),
                 transform=axes[0].get_yaxis_transform(), ha="right", va="center",
                 fontsize=7.4, color="#a4b0bc" if not prov else "#c08552")

handles = [Line2D([], [], color=D.RULE_COLOR[D.rule_of(c)], marker=D.CODE_MARK[D.code_of(c)],
                  ls="none", ms=8.5, label=c) for c in D.COLUMNS]
handles += [Line2D([], [], color="#5b6b7b", marker="o", ls="none", ms=8.5, mfc="white",
                   mew=2, label="lone crossing (provisional)"),
            Line2D([], [], color="#93a1ae", marker="x", ls="none", ms=8, mew=2,
                   label="never reaches 1e-4 in this sweep")]
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9.6,
           bbox_to_anchor=(.5, .015))
fig.suptitle("C3 — the scoreboard: cost to reach |ΔR| ≤ 1e-4 (first sustained)",
             fontsize=17, fontweight="bold", color=INK, y=.962)
fig.text(.5, .885, "left: how many Fourier orders it takes · right: how many milliseconds "
         "that solve costs — the same ranking read two ways",
         ha="center", fontsize=10.5, color=MUTED)
fig.savefig(out("out_C3_scoreboard.png"), dpi=120)
print("wrote out_C3_scoreboard.png")
