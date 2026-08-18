# Visualization proposals

Candidate figures for the benchmark battery and for the `night_run_2` convergence
data — **proposals, not yet wired into the shipped plotters**. Everything here is
rendered from the real repository data: geometry comes straight out of
`benchmark/structures.py`, numbers straight out of
`benchmark/night_run_2/conv_results.json` and `benchmark/moose_reference.json`.
No figure restates a dimension or a result, so none of them can drift from the
thing it claims to draw.

Rendered proofs live in `figures/` (force-added past `benchmark/.gitignore`'s
`*.png`, so the proposals are reviewable without running anything).

```bash
python benchmark/viz_proposals/make_all.py      # regenerate all of figures/
GRCWA_CONV_RUN=benchmark/some_other_run python benchmark/viz_proposals/make_all.py
```

`GRCWA_VIZ_OUT` overrides the output directory; `GRCWA_CONV_RUN` picks a different
convergence snapshot.

## Part A — the structures

| plate | script | what it is |
|---|---|---|
| **S1** | `plot_s1_xsec.py` | Dimensioned x–z cut, mechanical-drawing style: per-panel scale with a λ bar, thickness dimensioned in a right margin with extension lines, beam glyph carrying the polarization, plus an x–y top view for the 2D cases. |
| **S2** | `plot_s2_iso.py` | Isometric cut-away unit cell whose three visible faces are *textured with the real rasterized mask* — top face = x–y pattern, front face = x–z cut, side face = y–z cut. Free-standing cases get a phantom half-space instead of a solid substrate, and **E** is drawn along the axis the polarization selects, so TE-along-the-grooves vs TM-across-them reads as geometry. |
| **S3** | `plot_s3_atlas.py` | The atlas: all 13 on one shared length scale, sorted by period. Every row is the same 3.3 λ window, so how many periods you see *is* the sub-wavelength/diffracting story. |

Shared code: `palette.py` (materials, colours, physics badges, geometry
extraction), `draw_xsec.py`, `draw_iso.py`.

The proposed rule is **dimensionality-adaptive**, because the three cases differ
in what a *complete* description is — a cut through a square pillar and a cut
through a cylinder are the same drawing:

* **0D** (`A1`, `A1b`) — layer-stack column; there is nothing to cut.
* **1D** (`A2*`, `B1*`, `B2`, `B3`, `D1`) — the S1 x–z cut, which is already complete.
* **2D** (`C1`, `C1b`, `C2`, `D2`) — the S2 isometric, which carries the top view
  and the cross-section on its own faces instead of needing a second panel.
* **all 13** — S3 once, as the opener.

## Part B — the convergence

| plate | script | what it is |
|---|---|---|
| **C1 / C1t** | `plot_c1_digitmap.py` | The digit map: one row per code+rule, one colour step per correct digit of R, ▼ at the first sustained \|ΔR\| ≤ 1e-4. `C1` puts retained orders on x, `C1t` puts wall time on x. A rule that never turns yellow never converges. |
| **C2** | `plot_c2_staircase.py` | The cost staircase: raw measurements plus the running-best envelope — the honest answer to “with this time budget, how close can I get?”. Monotone by construction, which is what makes it readable where `conv_accuracy_vs_time.png` is not. |
| **C3** | `plot_c3_scoreboard.py` | The scoreboard: cost to reach 1e-4 for every case × column, in orders *and* in milliseconds. Hollow marker = lone crossing (provisional), ✗ = never reached. `conv_convergence.csv` as one picture. |
| **C4** | `plot_c4_moose.py` | Moose redesigned: signed deviation from the independent reference on a **symlog** axis, so the low-order transient and the 1e-4 endgame fit in one panel — one figure in place of `moose_compare_raw` + `_tight`. Plus a summary strip of how far each code still is at the end of the sweep. |

Shared code: `conv_data.py` (tidy arrays, references, Moose sweep parsing, the
running-min envelope, and the sustained-arrival rule that matches `conv_run.py`).

### Colour semantics — the one deliberate break with the existing plotters

`plot_conv.py` and `plot_moose.py` encode **colour = codebase, linestyle = rule**.
These proposals do the opposite: **colour = factorization rule, dash + marker =
codebase**, on the argument that the story this benchmark exists to tell is
direct-versus-faithful, not fork-versus-ikarus. A consequence worth stating
explicitly: both Laurent columns collapse onto the same grey, which *is* a claim —
that the direct rule behaves identically in both codes. The night run supports it
(they agree to ~1e-6, limited only by the `FREQC` detuning), but it is a claim, and
it should be a deliberate choice rather than a side effect.

Palette is Okabe–Ito, so the figures survive colour-blind readers and greyscale
printing; the current red/green encoding does not.

## Part X — the interactive explorer

`export_web.py` writes `figures/conv_web.json` (~46 kB): per case, per column, the
order grid, R, \|ΔR\| and the modelled solve time. That is the whole payload behind
the live explorer — case selector, y-axis switch (raw R / signed symlog / \|ΔR\| /
correct digits), x-axis switch (orders / wall time), per-series toggles — so the
same data can be re-read along whichever axis answers the question in front of you,
without regenerating a PNG.
