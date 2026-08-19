# Visualization: what shipped, and what is still a proposal

This directory is the review workspace for the branch's figure work. Two of the
three structure plates and two of the four convergence plates were adopted and
now live in `benchmark/` as normal plotters; what is left here is the rest,
plus a committed gallery so every figure is reviewable without running anything.

```bash
python benchmark/viz_proposals/make_all.py     # renders everything into figures/
```

`figures/` holds the rendered output of *both* the shipped plotters and the
remaining proposals. `GRCWA_CONV_JSON` picks a different convergence run
(default: `benchmark/conv_results.json`, falling back to the committed
`night_run_2` snapshot).

## Adopted — now in `benchmark/`

| figure | plotter | what it is |
|---|---|---|
| `struct_iso.png` | `plot_structures.py` | Every structure as a solid isometric unit cell. Air is absent rather than grey: gaps between ridges and pillars are open space, a perforated film is cut through to its substrate, free-standing cases get a dashed phantom half space. **E** is drawn along the axis the polarization selects. |
| `struct_atlas.png` | `plot_structures.py` | All 13 on one shared length scale, sorted by period — how many periods fit in the same 3.3 λ window *is* the sub-λ/diffracting story. |
| `conv_cost_staircase.png` | `plot_conv_cost.py` | The best \|ΔR\| a wall-time budget can buy: raw measurements plus the running-best step. |
| `conv_deviation_orders.png`, `conv_deviation_time.png` | `plot_deviation.py` | Signed deviation from the reference on a symlog axis, against retained orders and against wall time. Moose overlaid on the order variant. |

Shared code moved to `benchmark/` with them: `viz_palette.py` (materials,
colours, physics badges, geometry pulled from `structures.py`), `viz_iso.py`
(the isometric solid renderer) and `viz_conv.py` (tidy arrays, references, Moose
sweep parsing, the running-min envelope and the sustained-arrival rule that
matches `conv_run.py`).

## Still proposals

| plate | script | status |
|---|---|---|
| **S1** — dimensioned x–z cross-section | `plot_s1_xsec.py`, `draw_xsec.py` | Kept for rework. The λ bar, the margin dimension lines and the top-view inset all work, but several dimensions are still hard to read at sheet scale. Not wired into any plotter. |
| **C1 / C1t** — the digit map | `plot_c1_digitmap.py` | Correct digits of R as a heat strip, over retained orders and over wall time. Fast to read, but it throws away the sign and the shape of the decay. |
| **C3** — the scoreboard | `plot_c3_scoreboard.py` | Cost to reach 1e-4 for every case × column, in orders *and* in milliseconds; `conv_convergence.csv` as one picture. |

## `plate_book.html`

A self-contained review page: every plate side by side, click-to-zoom, plus an
interactive explorer over the whole run — case selector, y-axis switch (raw R /
signed symlog / \|ΔR\| / correct digits), x-axis switch (orders / wall time),
per-series toggles. `export_web.py` writes the ~46 kB payload it embeds
(`figures/conv_web.json`).

It is a one-off review tool by decision, not a build output: no HTML export is
wired into the shipped plotters.
