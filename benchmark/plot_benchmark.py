"""Plot the single-nG benchmark from results.csv.

Column-agnostic: it discovers whatever suites are present in results.csv (the
auto-discovered grcwa* variants + fork, each [Laurent] and optionally [Pol], plus
ikarus[Laurent]/[Li]/[NV] when Ikarus is installed) and assigns colors
automatically. Run benchmark/run.py first; then
`python benchmark/plot_benchmark.py` writes the figures into this folder.
"""
import os
import csv
import cmath
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'results.csv')
OUT = HERE

rows = []
with open(CSV) as f:
    for row in csv.DictReader(f):
        rows.append({
            'case': row['case'], 'column': row['column'],
            'R': float(row['R']), 'T': float(row['T']), 'A': float(row['A']),
            'nG': int(row['nG']), 'time_ms': float(row['time_ms']), 'mode': row['mode'],
        })

cases = list(dict.fromkeys(r['case'] for r in rows))
columns_seen = list(dict.fromkeys(r['column'] for r in rows))
data = defaultdict(dict)
for r in rows:
    data[r['column']][r['case']] = r

# Direct-rule (Laurent) columns first, in discovery order, then every faithful
# rule: Pol (grcwa), Li and NV (Ikarus). Anything unrecognized still gets a slot.
laurent_cols = [c for c in columns_seen if c.endswith('[Laurent]')]
pol_cols = [c for c in columns_seen if c.endswith('[Pol]')]
faithful_cols = [c for c in columns_seen if c not in laurent_cols]
col_order = laurent_cols + faithful_cols

# colors: one per column from a qualitative colormap; faithful columns get a hatch.
_palette = plt.get_cmap('tab10').colors + plt.get_cmap('Set2').colors
col_colors = {c: _palette[i % len(_palette)] for i, c in enumerate(col_order)}
col_hatch = {c: '//' for c in faithful_cols}
_marker_cycle = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', '<', '>']
col_markers = {c: _marker_cycle[i % len(_marker_cycle)] for i, c in enumerate(col_order)}

case_labels = {
    'A1_slab_air':             'A1 0D slab (air)',
    'A1b_slab_glass':          'A1b 0D slab (glass)',
    'A2_formbiref_TE':         'A2 form-biref TE',
    'A2_formbiref_TM':         'A2 form-biref TM',
    'B1_Si_grating_TE':        'B1 1D Si TE',
    'B1_Si_grating_TM':        'B1 1D Si TM',
    'B2_HCG_TM':               'B2 1D HCG TM',
    'B3_Au_slits_TM':          'B3 1D Au slits TM',
    'C1_Si_pillars':           'C1 2D Si pillars',
    'C1b_Si_pillars_diffract': 'C1b 2D Si diffract',
    'C2_Au_holes':             'C2 2D Au holes',
    'D1_ikarus_hcg_TM':        'D1 ikarus HCG TM',
    'D2_ikarus_cylinder_TE':   'D2 ikarus cylinder TE',
}
xlabels = [case_labels.get(c, c) for c in cases]
n_cases = len(cases); n_cols = len(col_order)
group_w = 0.86; bar_w = group_w / max(n_cols, 1)

def grouped_bars(ax, valfn, log=False):
    for ci, col in enumerate(col_order):
        xs, ys = [], []
        for ki, case in enumerate(cases):
            if case in data[col]:
                xs.append(ki + (ci - n_cols/2 + 0.5)*bar_w)
                ys.append(valfn(data[col][case]))
        if xs:
            ax.bar(xs, ys, width=bar_w*0.92, color=col_colors[col], label=col,
                   alpha=0.9, edgecolor='k', linewidth=0.4,
                   hatch=col_hatch.get(col))
    ax.set_xticks(range(n_cases)); ax.set_xticklabels(xlabels, fontsize=7.5, rotation=35, ha='right')
    if log: ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

# ===== Figure 1: R / T / A =====================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
fig.suptitle('RCWA benchmark  (lambda = 1 um)   -   direct vs faithful factorization',
             fontsize=13, fontweight='bold')
for ax, q, title, ylim in zip(
        axes, ['R', 'T', 'A'],
        ['Reflectance R', 'Transmittance T', 'Absorption A = 1-R-T'],
        [(0, 1.02), (0, 1.02), (-0.02, 0.35)]):
    grouped_bars(ax, lambda r, q=q: r[q])
    if q == 'A':
        ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel(title); ax.set_ylim(*ylim); ax.set_title(title)
axes[0].legend(fontsize=6.4, ncol=1, loc='upper right')
plt.tight_layout()
plt.savefig(f'{OUT}/bench_RTA.png', dpi=150, bbox_inches='tight'); plt.close()

# ===== Figure 2: timing ========================================================
fig2, ax2 = plt.subplots(figsize=(12, 5))
fig2.suptitle('Wall-time per case and suite  (min of 3 runs, log scale)',
              fontsize=12, fontweight='bold')
grouped_bars(ax2, lambda r: r['time_ms'], log=True)
ax2.set_ylabel('Wall time  [ms]'); ax2.legend(fontsize=7.5, ncol=2)
plt.tight_layout()
plt.savefig(f'{OUT}/bench_timing.png', dpi=150, bbox_inches='tight'); plt.close()

# ===== Figure 3: agreement cross-checks + energy ===============================
fig3, axes3 = plt.subplots(1, 4, figsize=(25, 5.4))
fig3.suptitle('Cross-checks  (lambda = 1 um)', fontsize=12, fontweight='bold')

def spread_panel(ax, cols, title):
    spreads = []
    for case in cases:
        rv = [data[c][case]['R'] for c in cols if case in data[c]]
        spreads.append(max(rv) - min(rv) if len(rv) > 1 else 0.0)
    ax.bar(range(len(cases)), [max(s, 1e-17) for s in spreads],
           color=['#c44e52' if s > 1e-6 else '#55a868' for s in spreads],
           edgecolor='k', linewidth=0.5)
    ax.set_yscale('log'); ax.set_ylim(1e-17, 1e-1)
    ax.set_xticks(range(len(cases))); ax.set_xticklabels(xlabels, fontsize=7.5, rotation=35, ha='right')
    ax.set_ylabel('max|dR| across columns')
    ax.set_title(title)
    ax.axhline(1e-6, color='orange', ls='--', lw=1.2, label='1e-6')
    ax.axhline(1e-12, color='green', ls=':', lw=1.2, label='1e-12')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3, which='both')

# (a) agreement among the grcwa family's Laurent columns -- same rule, same
#     conventions, so anything above the 1e-12 line is a real version difference.
grcwa_laurent = [c for c in laurent_cols if not c.startswith('ikarus')]
spread_panel(axes3[0], grcwa_laurent,
             f'grcwa Laurent agreement\n({len(grcwa_laurent)} versions)')
# (b) the same rule across INDEPENDENT codebases (grcwa vs Ikarus). Its floor is
#     ~1e-6, not 0: Ikarus takes a real frequency where this battery feeds grcwa
#     the Q=1e7 complex FREQC, so do not read green here.
cross_code = ([c for c in ('fork[Laurent]',) if c in data] +
              [c for c in laurent_cols if c.startswith('ikarus')])
if len(cross_code) > 1:
    spread_panel(axes3[1], cross_code,
                 'cross-code Laurent agreement\n(fork vs ikarus; floor ~1e-6)')
else:
    axes3[1].axis('off')
    axes3[1].set_title('cross-code Laurent agreement\n(needs ikarus-rcwa)')
# (c) agreement among all Pol columns (port faithfulness)
if len(pol_cols) > 1:
    spread_panel(axes3[2], pol_cols,
                 f'Pol agreement\n({len(pol_cols)} codebases)')
else:
    axes3[2].axis('off')
    axes3[2].set_title('Pol agreement\n(only one Pol column)')

# (d) energy conservation R+T
ax = axes3[3]
for ci, col in enumerate(col_order):
    xs, ys = [], []
    for ki, case in enumerate(cases):
        if case in data[col]:
            xs.append(ki + (ci - n_cols/2 + 0.5)*bar_w)
            ys.append(data[col][case]['R'] + data[col][case]['T'])
    if xs:
        ax.scatter(xs, ys, s=42, marker=col_markers.get(col, 'o'),
                   color=col_colors[col], label=col, zorder=3, alpha=0.95,
                   edgecolor='k', linewidth=0.3)
ax.axhline(1.0, color='k', lw=1.0, ls='--')
ax.set_xticks(range(n_cases)); ax.set_xticklabels(xlabels, fontsize=7.5, rotation=35, ha='right')
ax.set_ylabel('R + T'); ax.set_ylim(0.45, 1.05)
ax.set_title('Energy R+T (lossless -> 1; the faithful rules are not\n'
             'energy-conserving at finite order)')
ax.legend(fontsize=6.0, ncol=2, loc='lower left'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/bench_crosscheck.png', dpi=150, bbox_inches='tight'); plt.close()

# ===== Figure 4: direct (Laurent) vs faithful reflectance ======================
fig4, ax4 = plt.subplots(figsize=(12, 5.2))
fig4.suptitle('Direct (Laurent) vs faithful-factorization reflectance',
              fontsize=12, fontweight='bold')
rep_laurent = 'fork[Laurent]' if 'fork[Laurent]' in laurent_cols else (
    laurent_cols[0] if laurent_cols else None)
series = []
if rep_laurent:
    series.append((f'Laurent ({rep_laurent}; all agree)', rep_laurent, None))
for c in faithful_cols:
    series.append((c, c, '//'))
gw = 0.8; bw = gw / max(len(series), 1)
for ci, (lbl, col, hatch) in enumerate(series):
    xs, ys = [], []
    for ki, case in enumerate(cases):
        if case in data[col]:
            xs.append(ki + (ci - len(series)/2 + 0.5)*bw)
            ys.append(data[col][case]['R'])
    ax4.bar(xs, ys, width=bw*0.9, color=col_colors.get(col, '#888'), label=lbl,
            alpha=0.9, edgecolor='k', linewidth=0.4, hatch=hatch)
ax4.set_xticks(range(n_cases)); ax4.set_xticklabels(xlabels, fontsize=8, rotation=35, ha='right')
ax4.set_ylabel('Reflectance R'); ax4.set_ylim(0, 1.02)
ax4.legend(fontsize=8); ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/bench_direct_vs_faithful.png', dpi=150, bbox_inches='tight'); plt.close()

print("columns:", col_order)
if len(pol_cols) > 1:
    print("Pol agreement |dR| per case:")
    for case in cases:
        rv = [data[c][case]['R'] for c in pol_cols if case in data[c]]
        if len(rv) > 1:
            print(f"  {case:24s} {max(rv)-min(rv):.2e}")
print("figures written to", OUT)
