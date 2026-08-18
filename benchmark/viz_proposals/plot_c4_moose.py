"""C4 -- Moose redesign: signed deviation from the independent reference on a
symlog axis, so the wild low-order transient AND the 1e-4 endgame fit in ONE
panel (today that needs conv_raw + conv_raw_tight side by side)."""
import os

import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import conv_data as D

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.environ.get("GRCWA_VIZ_OUT", os.path.join(HERE, "figures"))
os.makedirs(OUTDIR, exist_ok=True)


def out(name):
    return os.path.join(OUTDIR, name)



plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                     "savefig.facecolor": "white"})
INK, MUTED = "#1b2733", "#5b6b7b"
LT = 1e-4
CASES = [c for c in D.ORDER if D.moose_series(c)[0] is not None]


def panel(ax, case):
    Rref, rt, prov = D.ref_of(case)
    for col in D.COLUMNS:
        nG, R, e, s, t, te = D.series(case, col)
        st = D.style(col)
        ax.plot(nG, s, color=st["color"], ls=st["linestyle"], marker=st["marker"],
                ms=3.4, lw=1.7, zorder=5, mec="none", alpha=.95)
    mx, mR, mref = D.moose_series(case)
    ax.plot(mx, mR - Rref, color=D.MOOSE_C, ls="-", marker="D", ms=5.0, lw=2.4,
            zorder=7, label="Moose", mfc="white", mew=1.4)
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=LT, linscale=.85)
    ax.axhspan(-LT, LT, color="#e6f4ec", zorder=1)
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
    return prov


fig = plt.figure(figsize=(17.5, 12.2))
gs = fig.add_gridspec(4, 3, hspace=.50, wspace=.20, left=.055, right=.985,
                      top=.878, bottom=.115)
for i, case in enumerate(CASES):
    ax = fig.add_subplot(gs[i // 3, i % 3])
    prov = panel(ax, case)
    ax.set_title(case + ("   (provisional Moose ref)" if prov else ""), fontsize=10.5,
                 fontweight="bold", color=INK, loc="left", pad=6)
    if i % 3 == 0:
        ax.set_ylabel("R − R_ref   (symlog)", fontsize=8.8, color=MUTED)
    if i >= len(CASES) - 3:
        ax.set_xlabel("total retained orders", fontsize=8.8, color=MUTED)

# ---- summary strip: agreement with the independent reference at top order ---
axs = fig.add_subplot(gs[3, :])
w = .15
for i, case in enumerate(CASES):
    for j, col in enumerate(D.COLUMNS):
        nG, R, e, s, t, te = D.series(case, col)
        if not len(e):
            continue
        st = D.style(col)
        axs.bar(i + (j - 2) * w, max(e[-1], 1e-9), width=w * .88, color=st["color"],
                alpha=.92, zorder=3)
axs.set_yscale("log")
axs.axhline(1e-4, color="#2e9e6b", lw=1.3, ls=(0, (4, 3)), zorder=4)
axs.text(len(CASES) - .45, 1.15e-4, "1e-4", color="#2e9e6b", fontsize=8.6,
         fontweight="bold", va="bottom", ha="right")
axs.set_xticks(range(len(CASES)))
axs.set_xticklabels(CASES, fontsize=9, rotation=12, ha="right")
axs.set_ylabel("|R − R_ref| at the\nhighest order run", fontsize=8.8, color=MUTED)
axs.set_title("how far each code still is from the independent reference at the end of the sweep",
              fontsize=10.5, fontweight="bold", color=INK, loc="left", pad=6)
axs.grid(axis="y", color="#eef2f6", lw=.7, zorder=0)
axs.set_axisbelow(True)
for sp in ("top", "right"):
    axs.spines[sp].set_visible(False)
axs.tick_params(labelsize=8.5)

handles = [Line2D([], [], color=D.RULE_COLOR[D.rule_of(c)], ls=D.CODE_DASH[D.code_of(c)],
                  marker=D.CODE_MARK[D.code_of(c)], ms=5, lw=1.8, label=c)
           for c in D.COLUMNS]
handles += [Line2D([], [], color=D.MOOSE_C, marker="D", ms=6, lw=2.4, mfc="white",
                   mew=1.4, label="Moose (independent reference code)"),
            Line2D([], [], color="#e6f4ec", lw=9, label="±1e-4 band (linear zone)")]
fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=9.6,
           bbox_to_anchor=(.5, .012))
fig.suptitle("C4 — Moose comparison, redesigned: signed deviation on a symlog axis",
             fontsize=17, fontweight="bold", color=INK, y=.975)
fig.text(.5, .928, "above the band you read the transient and the sign of the approach; inside "
         "the band the axis goes linear, so the endgame is legible in the same panel\n"
         "— one figure instead of today's conv_raw + conv_raw_tight pair",
         ha="center", fontsize=10.3, color=MUTED, linespacing=1.5)
fig.savefig(out("out_C4_moose_symlog.png"), dpi=112)
print("wrote out_C4_moose_symlog.png")
