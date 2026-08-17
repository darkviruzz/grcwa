"""Overlay the convergence-study results (benchmark/conv_results.json) with the
external 'Moose' reference (benchmark/moose_reference.json).

This does NOT recompute anything -- it plots the values already produced by
conv_run.py plus the Moose reference. Run conv_run.py first, then
`python benchmark/plot_moose.py`.

Which conv_results columns are drawn is controlled by WHITELIST below:
  * an entry with a rule, e.g. "grcwaProjects[Laurent]", matches that column only;
  * an entry without "[...]", e.g. "fork", matches ALL variants of that codebase
    (both [Laurent] and [Pol]).
Set WHITELIST = None to draw every column present. Moose is always overlaid.
Only cases Moose actually ran are drawn (the group-D factorization cases are not
among them -- they are referenced against Ikarus instead, in plot_conv.py).

x-axis = total retained orders. Moose 2D keys "(m,m)" are read as m per-axis
orders -> m*m total (matching the (q,q) convention in structures.py); 1D keys
"N" -> N. If Moose actually means max-order m (2m+1 per axis), change parse_key.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- which conv_results columns to draw ------------------------------------
WHITELIST = ["grcwaProjects[Laurent]", "weiliang-013[Pol]", "fork"]
# WHITELIST = None   # <- draw ALL columns present in conv_results
# WHITELIST = ["orig-0.1.2", "weiliang-013", "grcwaProjects", "codex",
#              "original-grcwaProjects", "fork"]   # every variant, both rules

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = HERE
FLOOR = 1e-7

CONV = json.load(open(os.path.join(HERE, "conv_results.json")))
MOOSE = json.load(open(os.path.join(HERE, "moose_reference.json")))
conv_cases = CONV["cases"]
moose_cases = MOOSE["cases"]
all_cols = CONV["columns"]


def selected(col):
    if WHITELIST is None:
        return True
    for spec in WHITELIST:
        if "[" in spec:
            if col == spec:
                return True
        elif col.split("[")[0] == spec:
            return True
    return False


COLS = [c for c in all_cols if selected(c)]

# ---- styling: colour = codebase, linestyle = rule; Moose = black -----------
_codes = list(dict.fromkeys(c.split("[")[0] for c in COLS))
_palette = ["#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#e377c2",
            "#7f7f7f", "#bcbd22"]
_code_color, _i = {}, 0
for code in _codes:
    if code == "fork":
        _code_color[code] = "#d62728"
    elif code.startswith("weiliang"):
        _code_color[code] = "#1f77b4"
    else:
        _code_color[code] = _palette[_i % len(_palette)]
        _i += 1


# solid = the direct (Laurent) rule; every faithful rule gets its own broken style.
_RULE_STYLE = {"Laurent": ("-", "o"), "Pol": ("--", "s"),
               "Li": ("-.", "^"), "NV": (":", "D")}


def style(col):
    color = _code_color.get(col.split("[")[0], "#555")
    rule = col.split("[")[-1].rstrip("]") if "[" in col else "Laurent"
    ls, mk = _RULE_STYLE.get(rule, ("--", "x"))
    return color, ls, mk


def parse_total(key):
    key = key.strip()
    if key.startswith("("):
        a, b = key.strip("()").split(",")
        m = int(a)
        n = int(b)
        m = 2 * m + 1       # moose uses m as maximum order symmetrically → 2m+1
        n = 2 * n + 1
        return m * n      # m x m square block -> m*n total
    n = int(key)
    n = 2 * n + 1
    return n                         # 1D: n orders


def conv_points(case, col):
    """sorted [(nG, R)] for a conv column."""
    pts = [(p["nG"], p["R"]) for p in conv_cases[case]["columns"].get(col, [])
           if "R" in p]
    return sorted(pts)


def moose_points(case):
    if case not in moose_cases:
        return []
    return sorted((parse_total(k), v) for k, v in moose_cases[case]["sweep"].items())


def ref_of(case):
    """Moose reference if present, else highest-order value of a selected
    Laurent column."""
    if case in moose_cases and moose_cases[case].get("ref") is not None:
        return moose_cases[case]["ref"]
    best = None
    for col in COLS:
        if col.endswith("[Laurent]"):
            pts = conv_points(case, col)
            if pts:
                best = pts[-1][1]
    if best is not None:
        return best
    mp = moose_points(case)
    return mp[-1][1] if mp else None


# cases that have a real sweep in at least one selected column
def has_sweep(case):
    if len(moose_points(case)) >= 2:
        return True
    return any(len(conv_points(case, c)) >= 2 for c in COLS)


# This figure is *about* the Moose reference, so a case Moose never ran has
# nothing to be compared against here -- drawing it would silently fall back to
# a high-order Laurent value as the "reference", which on the group-D
# factorization cases is not converged and would be actively misleading. Those
# cases have their own reference in ikarus_reference.json; see plot_conv.py.
cases = [c for c in conv_cases if conv_cases[c]["info"].get("dim", 2) != 0
         and c in moose_cases and has_sweep(c)]

# ---- console table ---------------------------------------------------------
print("WHITELIST ->", COLS if WHITELIST else "ALL columns")
print("\n%-24s %10s   %s" % ("case", "ref(Moose)", "selected columns @ highest order"))
for case in cases:
    ref = ref_of(case)
    parts = []
    for col in COLS:
        pts = conv_points(case, col)
        if pts:
            parts.append("%s=%.5f" % (col, pts[-1][1]))
    print("%-24s %10.5f   %s" % (case, ref if ref is not None else float("nan"),
                                 "  ".join(parts)))

# ---- figures ---------------------------------------------------------------
ncol = 3
nrow = int(np.ceil(len(cases) / ncol))


def grid(kind):
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 4.1 * nrow),
                             squeeze=False)
    for i, case in enumerate(cases):
        ax = axes[i // ncol][i % ncol]
        ref = ref_of(case)
        # conv columns
        for col in COLS:
            pts = conv_points(case, col)
            if len(pts) < 1:
                continue
            c, lstyle, mk = style(col)
            x = [p[0] for p in pts]
            if kind == "error":
                y = [max(abs(p[1] - ref), FLOOR) for p in pts]
                ax.loglog(x, y, ls=lstyle, marker=mk, color=c, ms=4, lw=1.5,
                          label=col, alpha=0.9)
            else:
                ax.semilogx(x, [p[1] for p in pts], ls=lstyle, marker=mk, color=c,
                            ms=4, lw=1.5, label=col, alpha=0.9)
        # Moose
        mp = moose_points(case)
        if mp:
            mx = [p[0] for p in mp]
            if kind == "error":
                ax.loglog(mx, [max(abs(p[1] - ref), FLOOR) for p in mp], "-D",
                          color="k", ms=4, lw=1.8, label="Moose", zorder=5)
            else:
                ax.semilogx(mx, [p[1] for p in mp], "-D", color="k", ms=4, lw=1.8,
                            label="Moose", zorder=5)
        if kind != "error" and ref is not None:
            ax.axhline(ref, color="k", ls="--", lw=0.8, alpha=0.6)
        prov = ""
        if case in moose_cases and moose_cases[case].get("ref_provisional"):
            prov = "  (prov. ref)"
        ax.set_title(case + prov, fontsize=9)
        ax.set_xlabel("total retained orders", fontsize=8)
        ax.set_ylabel("|R - R_ref|" if kind == "error" else "R", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    for j in range(len(cases), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    return fig


f1 = grid("error")
f1.suptitle("conv_results vs Moose: |R(N) - R_ref|   (x = total retained orders)",
            fontsize=13, fontweight="bold")
f1.tight_layout()
f1.savefig(f"{OUT}/moose_compare_error.png", dpi=150, bbox_inches="tight")

f2 = grid("raw")
f2.suptitle("conv_results vs Moose: raw R   (black = Moose; dashed line = reference)",
            fontsize=13, fontweight="bold")
f2.tight_layout()
f2.savefig(f"{OUT}/moose_compare_raw.png", dpi=150, bbox_inches="tight")

print("\nfigures written to", OUT)
