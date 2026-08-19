"""Plot the grcwa convergence study from conv_results.json.

Run benchmark/conv_run.py first (it writes conv_results.json next to this
script); then `python benchmark/plot_conv.py` writes the figures alongside it.

Reads the nested JSON (meta / columns / cases{info,ref,columns{col:[sweep]}})
and produces:
  1. error-decay grid     |R(nG) - R_ref| vs nG          (log-log)   -> the rate
  2. raw R(nG) grid        R vs nG with ref/analytic line (semilogx) -> settling
  3. raw R vs wall-time for every case                    (semilogx)
  4. grouped/smoothed timing vs retained orders            (log-log)
  5. accuracy vs wall-time for the headline cases         (log-log)
  6. 0D analytic anchors   |R - R_analytic|               (bars)
Style: color = codebase (red fork / blue weiliang / orange ikarus), linestyle =
factorization rule -- solid for the direct (Laurent) rule, and a distinct broken
style for each faithful one (-- Pol, -. Li, : NV). The two raw-R figures also
have a ``_tight`` copy limited to ``R_ref +/- 0.01`` for each case.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from timing_model import estimate_ms

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.environ.get("GRCWA_CONV_JSON", os.path.join(HERE, "conv_results.json"))
OUT = os.environ.get("GRCWA_PLOT_OUTPUT_DIR", HERE)
os.makedirs(OUT, exist_ok=True)
FLOOR = 1e-16
TIGHT_DELTA_R = 0.01

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
    """Return each column's points sorted by total retained order count."""
    out = {}
    for col in columns:
        sw = sorted((p for p in cases[case]["columns"].get(col, []) if "R" in p),
                    key=lambda p: p["nG"])
        if len(sw) >= 2:
            out[col] = sw
    return out

def timed_sweeps(case):
    """{col: timed points sorted by nG}, including one-point 0D sweeps."""
    out = {}
    for col in columns:
        sw = sorted((p for p in cases[case]["columns"].get(col, [])
                     if "R" in p and p.get("time_ms", 0) > 0),
                    key=lambda p: p["nG"])
        if sw:
            out[col] = sw
    return out

def estimated_time(point):
    """Smoothed timing estimate, falling back to the raw measurement."""
    return point.get("time_est_ms") or point.get("time_ms")

sweep_cases = [c for c in cases if cases[c]["info"].get("dim", 2) != 0
               and sweeps(c)]
zerod_cases = [c for c in cases if cases[c]["info"].get("dim", 2) == 0]

def title(case):
    info = cases[case]["info"]
    return f"{case}\n{info.get('desc','')}"

def tight_R_limits(rref):
    """The raw-R axis window spanning reference +/- 0.01."""
    return rref - TIGHT_DELTA_R, rref + TIGHT_DELTA_R

# ---- Figure 1: error decay grid ---------------------------------------------
n = len(sweep_cases)
ncol = 3
nrow = int(np.ceil(n / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.0 * nrow),
                         squeeze=False)
fig.suptitle(
    "Convergence: |R(N) - R_ref| vs total retained orders   "
    "(solid = direct/Laurent; -- Pol, -. Li, : NV  |  "
    "red = fork, blue = weiliang, orange = ikarus)",
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
raw_title = fig.suptitle(
    "Raw reflectance R(nG) settling to the converged value "
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
for i, case in enumerate(sweep_cases):
    rref, _ = ref_R(case)
    ax = axes[i // ncol][i % ncol]
    lower, upper = tight_R_limits(rref)
    ax.set_ylim(lower, upper)
raw_title.set_text("Raw reflectance R(nG), tight view: R_ref +/- 0.01")
plt.tight_layout()
plt.savefig(f"{OUT}/conv_raw_R_tight.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- Figure 3: raw R vs wall-time (all cases) -------------------------------
timed_cases = [case for case in cases if timed_sweeps(case)]
if timed_cases:
    nt = len(timed_cases)
    ntcol = 3
    ntrow = int(np.ceil(nt / ntcol))
    timed_r = [p["R"] for case in timed_cases
               for sw in timed_sweeps(case).values() for p in sw]
    rspan = max(timed_r) - min(timed_r)
    rpad = 0.02 * max(rspan, 1.0)
    rlimits = min(0.0, min(timed_r) - rpad), max(1.0, max(timed_r) + rpad)
    fig, axes = plt.subplots(ntrow, ntcol,
                             figsize=(6.2 * ntcol, 4.0 * ntrow),
                             squeeze=False)
    time_R_title = fig.suptitle(
        "Raw reflectance R vs estimated wall-time for every convergence case "
        "(grouped monotonic model; black dashed = reference)",
        fontsize=13, fontweight="bold", y=0.995)
    legend_lines = {}
    for i, case in enumerate(timed_cases):
        ax = axes[i // ntcol][i % ntcol]
        for col, sw in timed_sweeps(case).items():
            c, ls, mk = style(col)
            wall_ms = [estimated_time(p) for p in sw]
            reflectance = [p["R"] for p in sw]
            line, = ax.semilogx(wall_ms, reflectance, ls=ls, marker=mk,
                                color=c, ms=4, lw=1.6, label=col, alpha=0.9)
            legend_lines.setdefault(col, line)
        rref, _ = ref_R(case)
        if rref is not None:
            ax.axhline(rref, color="k", ls="--", lw=1.0, alpha=0.7)
        ax.set_title(title(case), fontsize=8.5)
        ax.set_xlabel("estimated wall time per solve [ms]", fontsize=8)
        ax.set_ylabel("R", fontsize=8)
        ax.set_ylim(*rlimits)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
    for j in range(nt, ntrow * ntcol):
        axes[j // ntcol][j % ntcol].axis("off")
    legend_cols = [col for col in columns if col in legend_lines]
    fig.legend([legend_lines[col] for col in legend_cols], legend_cols,
               loc="upper center", bbox_to_anchor=(0.5, 0.98),
               fontsize=7, ncol=4)
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    plt.savefig(f"{OUT}/conv_R_vs_time.png", dpi=150, bbox_inches="tight")
    for i, case in enumerate(timed_cases):
        rref, _ = ref_R(case)
        ax = axes[i // ntcol][i % ntcol]
        lower, upper = tight_R_limits(rref)
        ax.set_ylim(lower, upper)
    time_R_title.set_text("Raw reflectance R vs estimated wall-time, "
                          "tight view: R_ref +/- 0.01")
    plt.tight_layout(rect=(0, 0, 1, 0.955))
    plt.savefig(f"{OUT}/conv_R_vs_time_tight.png", dpi=150,
                bbox_inches="tight")
    plt.close()

# ---- Figure 4: grouped and smoothed timing scaling --------------------------
timing_models = J.get("timing_models", [])
timing_dims = [dim for dim in (1, 2)
               if any(int(model.get("dim", -1)) == dim
                      for model in timing_models)]
if timing_dims:
    fig, axes = plt.subplots(1, len(timing_dims),
                             figsize=(7.0 * len(timing_dims), 5.0),
                             squeeze=False)
    fig.suptitle(
        "Measured solve time and grouped monotonic timing model "
        "(faint = structures; thick = smoothed estimate)",
        fontsize=12, fontweight="bold")
    for index, dim in enumerate(timing_dims):
        ax = axes[0][index]
        for model in timing_models:
            if int(model.get("dim", -1)) != dim:
                continue
            col = model["column"]
            c, ls, mk = style(col)
            raw_nG, raw_ms = [], []
            for case in cases.values():
                if int(case.get("info", {}).get("dim", -1)) != dim:
                    continue
                for point in case.get("columns", {}).get(col, []):
                    if point.get("time_ms", 0) > 0:
                        raw_nG.append(point["nG"])
                        raw_ms.append(point["time_ms"])
            if raw_nG:
                ax.loglog(raw_nG, raw_ms, linestyle="none", marker=mk,
                          color=c, ms=3, alpha=0.16)
            lo, hi = model["nG_min"], model["nG_max"]
            curve_nG = ([lo] if lo == hi else
                        np.geomspace(lo, hi, 160).tolist())
            curve_ms = [estimate_ms(model, nG) for nG in curve_nG]
            ax.loglog(curve_nG, curve_ms, ls=ls, color=c, lw=2.2,
                      label=col)
            anchors = model.get("samples", [])
            ax.loglog([sample["nG"] for sample in anchors],
                      [sample["fitted_time_ms"] for sample in anchors],
                      linestyle="none", marker=mk, color=c, ms=4)
        ax.set_title(f"{dim}D")
        ax.set_xlabel("total retained orders nG")
        ax.set_ylabel("wall time per solve [ms]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/conv_timing_model.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---- Figure 5: accuracy vs wall-time (headline cases) -----------------------
# D1 leads: it is the whitepaper's factorization case, and the one where the
# accuracy-per-second gap between the rules is widest.
headline = [c for c in ["D1_ikarus_hcg_TM", "B1_Si_grating_TM", "B2_HCG_TM",
                        "B3_Au_slits_TM", "C2_Au_holes"]
            if c in cases and sweeps(c)]
if headline:
    m = len(headline)
    fig, axes = plt.subplots(1, m, figsize=(5.2 * m, 4.6), squeeze=False)
    fig.suptitle(
        "Accuracy vs wall-time - does a faithful rule reach a target error faster?",
        fontsize=12, fontweight="bold")
    for i, case in enumerate(headline):
        ax = axes[0][i]
        rref, _ = ref_R(case)
        for col, sw in sweeps(case).items():
            c, ls, mk = style(col)
            t = [estimated_time(p) for p in sw if estimated_time(p)]
            err = [max(abs(p["R"] - rref), FLOOR)
                   for p in sw if estimated_time(p)]
            ax.loglog(t, err, ls=ls, marker=mk, color=c, ms=4, lw=1.6,
                      label=col, alpha=0.9)
        ax.set_title(title(case), fontsize=8.5)
        ax.set_xlabel("estimated wall time [ms]", fontsize=8)
        ax.set_ylabel("|R - R_ref|", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUT}/conv_accuracy_vs_time.png", dpi=150, bbox_inches="tight")
    plt.close()

# ---- Figure 6: 0D analytic anchors ------------------------------------------
if zerod_cases:
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.suptitle(
        "0D analytic anchors: |R_grcwa - R_Airy| (exact reference)",
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
