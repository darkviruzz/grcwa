"""Plot the grcwa convergence study from conv_results.json.

Run benchmark/conv_run.py first (it writes conv_results.json next to this
script); then `python benchmark/plot_conv.py` writes the figures alongside it.

Reads the nested JSON (meta / columns / cases{info,ref,columns{col:[sweep]}})
and produces:
  1. error-decay grid     |R(nG) - R_ref| vs nG          (log-log)   -> the rate
  2. raw R(nG) grid        R vs nG with ref/analytic line (semilogx) -> settling
  3. accuracy vs wall-time for the headline cases         (log-log)
  4. 0D analytic anchors   |R - R_analytic|               (bars)
Style: color = codebase (red fork / blue weiliang), linestyle = rule
(solid Laurent / dashed Pol).
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "conv_results.json")
OUT = HERE
FLOOR = 1e-16

with open(JSON) as f:
    J = json.load(f)

columns = J["columns"]
cases = J["cases"]

# one color per codebase (text before the [rule]); linestyle encodes the rule.
# fork stays red and weiliang* blue for continuity with the earlier plots; every
# other codebase takes the next free palette colour.
_codes = list(dict.fromkeys(c.split("[")[0] for c in columns))
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

# linestyle/marker encode the factorization rule: the direct (Laurent) rule is
# solid, every faithful rule is broken -- Pol (grcwa), Li's inverse rule and the
# normal-vector method (both Ikarus).
_RULE_STYLE = {"Laurent": ("-", "o"), "Pol": ("--", "s"),
               "Li": ("-.", "^"), "NV": (":", "D")}


def rule_of(col):
    return col.split("[")[-1].rstrip("]") if "[" in col else "Laurent"


def style(col):
    code = col.split("[")[0]
    color = _code_color.get(code, "#555")
    ls, mk = _RULE_STYLE.get(rule_of(col), ("--", "x"))
    return color, ls, mk

def ref_R(case):
    """Exact analytic if available, else an external converged reference, else
    the fork[Laurent] highest-order value.

    The external branch matters for the group-D factorization stress tests: there
    Laurent is still percent-level wrong at every order in the sweep, so its
    high-order value is not a usable reference (see conv_run.py)."""
    ref = cases[case].get("ref")
    if ref and ref.get("type") == "analytic_exact":
        return ref["R"], "analytic"
    if ref and str(ref.get("type", "")).startswith("external"):
        return ref["R"], ref.get("from") or ref["type"]
    sw = [p for p in cases[case]["columns"].get("fork[Laurent]", []) if "R" in p]
    if sw:
        return max(sw, key=lambda p: p["nG"])["R"], "self(fork Laurent, high order)"
    if ref:
        return ref["R"], ref.get("type", "ref")
    return None, None

def sweeps(case):
    """{col: points sorted by total order count nG}. 1D has q and q**2 points."""
    out = {}
    for col in columns:
        sw = sorted((p for p in cases[case]["columns"].get(col, []) if "R" in p),
                    key=lambda p: p["nG"])
        if len(sw) >= 2:
            out[col] = sw
    return out

sweep_cases = [c for c in cases if cases[c]["info"].get("dim", 2) != 0
               and sweeps(c)]
zerod_cases = [c for c in cases if cases[c]["info"].get("dim", 2) == 0]

def title(case):
    info = cases[case]["info"]
    return f"{case}\n{info.get('desc','')}"

# ---- Figure 1: error decay grid ---------------------------------------------
n = len(sweep_cases)
ncol = 3
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow),
                         squeeze=False)
fig.suptitle("Convergence: |R(N) - R_ref| vs total retained orders   "
             "(solid = Laurent, dashed = Pol; red = fork, blue = weiliang)",
             fontsize=13, fontweight="bold")
for i, case in enumerate(sweep_cases):
    ax = axes[i // ncol][i % ncol]
    rref, rkind = ref_R(case)
    for col, sw in sweeps(case).items():
        c, ls, mk = style(col)
        ng = [p["nG"] for p in sw]
        err = [max(abs(p["R"] - rref), FLOOR) for p in sw]
        ax.loglog(ng, err, ls=ls, marker=mk, color=c, ms=4, lw=1.6,
                  label=col, alpha=0.9)
    ax.set_title(title(case), fontsize=8.5)
    ax.set_xlabel("total retained orders", fontsize=8)
    ax.set_ylabel("|R(nG) - R_ref|", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=7)
for j in range(n, nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/conv_error_decay.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Figure 2: raw R(nG) grid -----------------------------------------------
fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow),
                         squeeze=False)
fig.suptitle("Raw reflectance R(nG) settling to the converged value "
             "(black dashed = reference; green dotted = analytic EMT where applicable)",
             fontsize=13, fontweight="bold")
for i, case in enumerate(sweep_cases):
    ax = axes[i // ncol][i % ncol]
    rref, rkind = ref_R(case)
    for col, sw in sweeps(case).items():
        c, ls, mk = style(col)
        ng = [p["nG"] for p in sw]
        rv = [p["R"] for p in sw]
        ax.semilogx(ng, rv, ls=ls, marker=mk, color=c, ms=4, lw=1.6,
                    label=col, alpha=0.9)
    if rref is not None:
        ax.axhline(rref, color="k", ls="--", lw=1.0, alpha=0.7)
    ref = cases[case].get("ref")
    if ref and ref.get("type") == "analytic_asymptotic":
        ax.axhline(ref["R"], color="green", ls=":", lw=1.3,
                   label=f"EMT n_eff={ref.get('n_eff', 0):.2f}")
    ax.set_title(title(case), fontsize=8.5)
    ax.set_xlabel("total retained orders", fontsize=8)
    ax.set_ylabel("R", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5)
for j in range(n, nrow * ncol):
    axes[j // ncol][j % ncol].axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/conv_raw_R.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Figure 3: accuracy vs wall-time (headline cases) -----------------------
headline = [c for c in ["B1_Si_grating_TM", "B2_HCG_TM", "B3_Au_slits_TM",
                        "C2_Au_holes"] if c in cases and sweeps(c)]
if headline:
    m = len(headline)
    fig, axes = plt.subplots(1, m, figsize=(5.2 * m, 4.6), squeeze=False)
    fig.suptitle("Accuracy vs wall-time - does Pol reach a target error faster?",
                 fontsize=12, fontweight="bold")
    for i, case in enumerate(headline):
        ax = axes[0][i]
        rref, _ = ref_R(case)
        for col, sw in sweeps(case).items():
            c, ls, mk = style(col)
            t = [p["time_ms"] for p in sw if p.get("time_ms")]
            err = [max(abs(p["R"] - rref), FLOOR) for p in sw if p.get("time_ms")]
            ax.loglog(t, err, ls=ls, marker=mk, color=c, ms=4, lw=1.6,
                      label=col, alpha=0.9)
        ax.set_title(title(case), fontsize=8.5)
        ax.set_xlabel("wall time [ms]", fontsize=8)
        ax.set_ylabel("|R - R_ref|", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/conv_accuracy_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---- Figure 4: 0D analytic anchors ------------------------------------------
if zerod_cases:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle("0D analytic anchors: |R_grcwa - R_Airy| (exact reference)",
                 fontsize=11, fontweight="bold")
    labels, errs = [], []
    for case in zerod_cases:
        ref = cases[case].get("ref")
        sw = [p for p in cases[case]["columns"].get("fork[Laurent]", [])
              if "R" in p]
        if ref and sw:
            labels.append(case)
            errs.append(max(abs(sw[-1]["R"] - ref["R"]), FLOOR))
    ax.bar(range(len(labels)), errs, color="#2ca02c", edgecolor="k")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|R - R_analytic|")
    ax.axhline(1e-12, color="green", ls=":", label="1e-12")
    ax.grid(axis="y", which="both", alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/conv_0D_anchors.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---- console summary --------------------------------------------------------
print("Convergence summary (error at highest nG vs reference):")
for case in sweep_cases:
    rref, rkind = ref_R(case)
    print(f"\n{case}  ref={rref:.6f} [{rkind}]")
    for col, sw in sweeps(case).items():
        e_lo = abs(sw[0]["R"] - rref)
        e_hi = abs(sw[-1]["R"] - rref)
        print(f"   {col:<24} nG {sw[0]['nG']:>4}->{sw[-1]['nG']:<4}"
              f"  |dR| {e_lo:.2e} -> {e_hi:.2e}")
print("\nfigures written to", OUT)
