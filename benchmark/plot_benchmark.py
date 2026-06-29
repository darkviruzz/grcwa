"""Plot the 5-suite single-nG benchmark from results.csv (6 columns).

Run benchmark/run.py first (it writes results.csv next to this script); then
`python benchmark/plot_benchmark.py` writes the figures into the same folder.
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

# fixed column order (Laurent group, then Pol group)
col_order = [
    'orig-0.1.2[Laurent]',
    'weiliang-0.1.3[Laurent]',
    'forkmaster[Laurent]',
    'fork[Laurent]',
    'weiliang-0.1.3[Pol]',
    'fork[Pol]',
]
col_order = [c for c in col_order if c in columns_seen]

def airy_R(n0, n1, ns, d, freq):
    k0 = 2*np.pi*freq
    r01 = (n0-n1)/(n0+n1); r12 = (n1-ns)/(n1+ns)
    ph = cmath.exp(2j*k0*n1*d)
    return abs((r01 + r12*ph)/(1 + r01*r12*ph))**2
airy = airy_R(1.0, 2.0, 1.0, 0.30, 1.0)

col_colors = {
    'orig-0.1.2[Laurent]':     '#4c72b0',   # blue
    'weiliang-0.1.3[Laurent]': '#55a868',   # green
    'forkmaster[Laurent]':     '#c44e52',   # red-ish
    'fork[Laurent]':           '#8172b3',   # purple
    'weiliang-0.1.3[Pol]':     '#ccb974',   # gold  (Pol)
    'fork[Pol]':               '#da8bc3',   # pink  (Pol)
}
col_hatch = {
    'weiliang-0.1.3[Pol]': '//',
    'fork[Pol]':           '\\\\',
}
col_markers = {
    'orig-0.1.2[Laurent]': 'o', 'weiliang-0.1.3[Laurent]': 's',
    'forkmaster[Laurent]': '^', 'fork[Laurent]': 'v',
    'weiliang-0.1.3[Pol]': 'D', 'fork[Pol]': 'P',
}
case_labels = {
    '2D_Si_hole_subwave':  '2D Si\nsubwave',
    '2D_Si_hole_diffract': '2D Si\ndiffract',
    '2D_metal_hole_absorb':'2D metal\nabsorb',
    '1D_Si_TE_subwave':    '1D Si\nTE sub',
    '1D_Si_TM_diffract':   '1D Si\nTM diff',
    '1D_metal_TM_absorb':  '1D metal\nTM abs',
    '0D_slab':             '0D slab',
}
xlabels = [case_labels.get(c, c) for c in cases]
n_cases = len(cases); n_cols = len(col_order)
group_w = 0.84; bar_w = group_w / n_cols

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
    ax.set_xticks(range(n_cases)); ax.set_xticklabels(xlabels, fontsize=7.5)
    if log: ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

# ===== Figure 1: R / T / A =====================================================
fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
fig.suptitle('grcwa 5-suite benchmark  (lambda = 1 um)   -   Laurent vs Pol factorization',
             fontsize=13, fontweight='bold')
for ax, q, title, ylim in zip(
        axes, ['R', 'T', 'A'],
        ['Reflectance R', 'Transmittance T', 'Absorption A = 1-R-T'],
        [(0, 0.75), (0, 1), (-0.02, 0.6)]):
    grouped_bars(ax, lambda r, q=q: r[q])
    if q == 'A':
        ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_ylabel(title); ax.set_ylim(*ylim); ax.set_title(title)
axes[0].legend(fontsize=6.6, ncol=1, loc='upper right')
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

# ===== Figure 3: Pol port faithfulness + Laurent agreement + energy ============
fig3, axes3 = plt.subplots(1, 3, figsize=(19, 5.2))
fig3.suptitle('Cross-checks  (lambda = 1 um)', fontsize=12, fontweight='bold')

# (a) Pol port faithfulness: weiliang-0.1.3[Pol] vs fork[Pol]
ax = axes3[0]
wl, fk = 'weiliang-0.1.3[Pol]', 'fork[Pol]'
dpol, lab, modes = [], [], []
for case in cases:
    if case in data[wl] and case in data[fk]:
        dpol.append(abs(data[wl][case]['R'] - data[fk][case]['R']))
        lab.append(case_labels.get(case, case))
        # native (fork) vs degenerate-2D (weiliang) => different nG for 1D
        same = data[wl][case]['nG'] == data[fk][case]['nG']
        modes.append(same)
colors = ['#55a868' if s else '#dd8452' for s in modes]
ax.bar(range(len(dpol)), [max(d, 1e-17) for d in dpol], color=colors,
       edgecolor='k', linewidth=0.5)
ax.set_yscale('log'); ax.set_ylim(1e-17, 1e-1)
ax.set_xticks(range(len(dpol))); ax.set_xticklabels(lab, fontsize=7.5)
ax.set_ylabel('|dR|  weiliang-0.1.3[Pol] - fork[Pol]')
ax.set_title('Pol port faithfulness\n(green = identical geometry/nG, orange = native-1D vs degen-2D)')
ax.axhline(1e-12, color='green', ls=':', lw=1.2, label='1e-12 (machine eps)')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3, which='both')

# (b) agreement among ALL Laurent columns
ax = axes3[1]
laurent_cols = [c for c in col_order if '[Laurent]' in c]
spreads = []
for case in cases:
    rv = [data[c][case]['R'] for c in laurent_cols if case in data[c]]
    spreads.append(max(rv)-min(rv) if len(rv) > 1 else 0.0)
ax.bar(range(len(cases)), [max(s, 1e-17) for s in spreads],
       color=['#c44e52' if s > 1e-6 else '#55a868' for s in spreads],
       edgecolor='k', linewidth=0.5)
ax.set_yscale('log'); ax.set_ylim(1e-17, 1e-1)
ax.set_xticks(range(len(cases))); ax.set_xticklabels(xlabels, fontsize=7.5)
ax.set_ylabel('max|dR| across Laurent columns')
ax.set_title('Laurent agreement\n(4 suites: orig / weiliang / forkmaster / fork)')
ax.axhline(1e-6, color='orange', ls='--', lw=1.2, label='1e-6')
ax.axhline(1e-12, color='green', ls=':', lw=1.2, label='1e-12')
ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3, which='both')

# (c) energy conservation R+T
ax = axes3[2]
for ci, col in enumerate(col_order):
    xs, ys = [], []
    for ki, case in enumerate(cases):
        if case in data[col]:
            xs.append(ki + (ci - n_cols/2 + 0.5)*bar_w)
            ys.append(data[col][case]['R'] + data[col][case]['T'])
    if xs:
        ax.scatter(xs, ys, s=45, marker=col_markers.get(col, 'o'),
                   color=col_colors[col], label=col, zorder=3, alpha=0.95,
                   edgecolor='k', linewidth=0.3)
ax.axhline(1.0, color='k', lw=1.0, ls='--')
ax.set_xticks(range(n_cases)); ax.set_xticklabels(xlabels, fontsize=7.5)
ax.set_ylabel('R + T'); ax.set_ylim(0.45, 1.05)
ax.set_title('Energy R+T (lossless -> 1; Pol breaks it for metal/diffract)')
ax.legend(fontsize=6.4, ncol=2, loc='lower left'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/bench_crosscheck.png', dpi=150, bbox_inches='tight'); plt.close()

# ===== Figure 4: Laurent vs Pol R, side by side (the physics story) ============
fig4, ax4 = plt.subplots(figsize=(12, 5.2))
fig4.suptitle('Laurent vs Pol reflectance - same algorithm in weiliang-0.1.3 and in the fork',
              fontsize=12, fontweight='bold')
pairs = [
    ('Laurent (all 4 suites agree)', 'fork[Laurent]', '#8172b3', None),
    ('Pol - weiliang-0.1.3', 'weiliang-0.1.3[Pol]', '#ccb974', '//'),
    ('Pol - fork (ported)', 'fork[Pol]', '#da8bc3', '\\\\'),
]
gw = 0.8; bw = gw/len(pairs)
for ci, (lbl, col, color, hatch) in enumerate(pairs):
    xs, ys = [], []
    for ki, case in enumerate(cases):
        if case in data[col]:
            xs.append(ki + (ci - len(pairs)/2 + 0.5)*bw)
            ys.append(data[col][case]['R'])
    ax4.bar(xs, ys, width=bw*0.9, color=color, label=lbl, alpha=0.9,
            edgecolor='k', linewidth=0.4, hatch=hatch)
ax4.set_xticks(range(n_cases)); ax4.set_xticklabels(xlabels, fontsize=8)
ax4.set_ylabel('Reflectance R'); ax4.set_ylim(0, 0.7)
ax4.legend(fontsize=9); ax4.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT}/bench_laurent_vs_pol.png', dpi=150, bbox_inches='tight'); plt.close()

print("Done. weiliang-0.1.3[Pol] vs fork[Pol] |dR| per case:")
for case in cases:
    if case in data[wl] and case in data[fk]:
        d = abs(data[wl][case]['R'] - data[fk][case]['R'])
        ng_w, ng_f = data[wl][case]['nG'], data[fk][case]['nG']
        print(f"  {case:22s} |dR|={d:.2e}  nG(wl)={ng_w} nG(fork)={ng_f}")
print("figures written to", OUT)
