# Plate book internals

The plate book is built with one command from the repository root:

```bash
python benchmark/make_plate_book.py      # -> benchmark/plate_book.html
```

That is the only entry point. This directory holds the parts it assembles;
nothing here needs to be run by hand.

> The directory name is historical. It began as a review workspace where figure
> ideas were parked next to adopted ones, and that distinction is gone: all
> seven plotters below are required, because the page template references a
> figure from each and the build refuses to inline a missing one.

## What runs, in order

`make_plate_book.py` renders these into `figures/`, then inlines the lot:

| step | script | figures |
|---|---|---|
| 1 | `benchmark/plot_structures.py` | `struct_iso.png`, `struct_atlas.png` |
| 2 | `benchmark/plot_conv_cost.py` | `conv_cost_staircase.png` |
| 3 | `benchmark/plot_deviation.py` | `conv_deviation_orders.png`, `conv_deviation_time.png` |
| 4 | `plot_s1_xsec.py` (+ `draw_xsec.py`) | `out_S1_crosssection.png` |
| 5 | `plot_c1_digitmap.py` | `out_C1_digitmap_orders.png`, `out_C1_digitmap_time.png` |
| 6 | `plot_c3_scoreboard.py` | `out_C3_scoreboard.png` |
| 7 | `export_web.py` | `conv_web.json` — the explorer payload plus the run's `meta` |
| 8 | `build_plate_book.py` | `plate_book.html` |

Templates live in `plate_book/`: `head.html` (title, fonts, stylesheet),
`body.html` (the content) and `app.html` (the explorer script). `body.html`
carries two kinds of placeholder, both resolved by `build_plate_book.py`:
`{{IMG:<name>.png}}` becomes a base64 data URI from `figures/`, and
`{{META:<KEY>}}` becomes a value measured from the run itself. `app.html`'s
`{{DATA}}` takes `conv_web.json` whole. The result is self-contained — it opens
from disk with no server, and the only external request is the Google Fonts
stylesheet in `head.html`.

An unknown `{{META:...}}` key fails the build rather than shipping a literal
placeholder into the page.

## Where the data comes from

Four inputs, one source each, no fallback. `make_plate_book.py` holds them as
editable strings at the top of the file and passes them down; a missing input
stops the build and names the command that produces it.

| input | produced by | consumed by |
|---|---|---|
| `benchmark/structures.py` | — (the battery itself) | `viz_palette.py` → the structure plates. Add a dict to `STRUCTURES` and it appears with no other edit. |
| `benchmark/conv_results.json` | `conv_run.py`, via `run_overnight.bat` | `viz_conv.py` → every convergence plate and the explorer |
| `benchmark/moose_reference.json` | `moose/moose_csv_to_json.py` from a Moose `moose_conv.csv` | the black Moose overlay, and the live reference for cases judged against Moose |
| `benchmark/moose_timing.json` | same converter, as a sibling file | the Moose points on the wall-time axis, on the C4 plate and in the explorer. Optional: without it Moose simply gets no cost axis. |

`GRCWA_SYMLOG_LINTHRESH` (default 1e-6) sets the half-width of the linear band
on the signed-deviation views. `viz_conv.linthresh` is the single reader:
`plot_deviation.py` scales its axis with it and `export_web.py` ships it in
`meta`, so the C4 plate and the explorer's symlog mode cannot drift apart.

Nothing is transcribed. Add data at the source and every figure follows.

## Reading a book built mid-sweep

`conv_run.py` rewrites `conv_results.json` at the end of every stage, so a book
built while `run_overnight.bat` is running is a snapshot of the last finished
stage. The header says so — it prints the build time, the q reached, the solve
count and whether the run reported itself complete, all measured at build time
rather than typed into the template.

Because that rewrite is not atomic, a build can catch the file mid-write.
`viz_conv._load` retries a truncated read a few times before giving up, so this
is normally invisible; if it does fail it says to wait for the stage to finish.
