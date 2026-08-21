"""Build the plate book: every figure, then the self-contained HTML page.

    python benchmark/make_plate_book.py

One command, one page.  It renders the seven plotters into ``FIGURE_DIR``,
exports the run as JSON for the interactive explorer, and inlines all of it into
``PLATE_BOOK`` -- a single HTML file that opens from disk.

Retargeting the book at a different run means editing the four INPUT strings
below; nothing else in the pipeline hard-codes a path.  There is deliberately no
fallback: if an input is missing the build stops and names the command that
produces it, because a plate that quietly plots an older run is worse than no
plate.

The book is a snapshot, not a live view.  Built while a sweep is running it
shows the last stage ``conv_run.py`` finished writing, and the header records
which one that was.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROPOSALS = os.path.join(HERE, "viz_proposals")

# ---------------------------------------------------------------------------
# INPUTS -- edit these to point the plate book at a different run.
# Defaults are where run_overnight.bat and moose_csv_to_json.py actually write.
# ---------------------------------------------------------------------------
CONV_JSON = os.path.join(HERE, "conv_results.json")
MOOSE_JSON = os.path.join(HERE, "moose_reference.json")
MOOSE_TIMING_JSON = os.path.join(HERE, "moose_timing.json")

# Half-width of the linear band on the symlog deviation plates.  Smaller buys
# resolution in the endgame and collapses everything below it into one flat
# zone; 1e-6 keeps the 1e-4 tolerance itself on the readable log side.
SYMLOG_LINTHRESH = "1e-6"

# ---------------------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------------------
FIGURE_DIR = os.path.join(PROPOSALS, "figures")
PLATE_BOOK = os.path.join(HERE, "plate_book.html")

# Plotters, in the order the book presents them.  All seven are required: the
# page template references a figure from each, and the build refuses to inline
# a missing one.
PLOTTERS = [
    (HERE, "plot_structures.py"),        # struct_iso, struct_atlas
    (HERE, "plot_conv_cost.py"),         # conv_cost_staircase
    (HERE, "plot_deviation.py"),         # conv_deviation_orders, _time
    (PROPOSALS, "plot_s1_xsec.py"),      # out_S1_crosssection
    (PROPOSALS, "plot_c1_digitmap.py"),  # out_C1_digitmap_orders, _time
    (PROPOSALS, "plot_c3_scoreboard.py"),  # out_C3_scoreboard
    (PROPOSALS, "export_web.py"),        # conv_web.json for the explorer
]


def _env():
    """Environment the plotters inherit: the inputs above, and one figure dir.

    The shipped plotters read GRCWA_PLOT_OUTPUT_DIR, the proposals read
    GRCWA_VIZ_OUT; both are pointed at FIGURE_DIR so the book collects one set.
    """
    return dict(
        os.environ,
        GRCWA_CONV_JSON=CONV_JSON,
        GRCWA_MOOSE_JSON=MOOSE_JSON,
        GRCWA_MOOSE_TIMING_JSON=MOOSE_TIMING_JSON,
        GRCWA_SYMLOG_LINTHRESH=SYMLOG_LINTHRESH,
        GRCWA_PLOT_OUTPUT_DIR=FIGURE_DIR,
        GRCWA_VIZ_OUT=FIGURE_DIR,
        GRCWA_VIZ_FIGURES=FIGURE_DIR,
        GRCWA_PLATE_BOOK=PLATE_BOOK,
        MPLBACKEND="Agg",
    )


def _check_inputs():
    """Fail before rendering anything, not after the slow isometric sheet."""
    missing = [(path, hint) for path, hint in [
        (CONV_JSON, "python benchmark/conv_run.py   (or benchmark/run_overnight.bat)"),
        (MOOSE_JSON, "python benchmark/moose/moose_csv_to_json.py <moose_conv.csv>"),
    ] if not os.path.exists(path)]
    if missing:
        lines = ["the plate book needs inputs that do not exist yet:"]
        for path, hint in missing:
            lines.append("  %s\n      produce it with: %s" % (path, hint))
        lines.append("Edit the INPUT strings at the top of %s to use a different run."
                     % os.path.basename(__file__))
        raise SystemExit("\n".join(lines))
    if not os.path.exists(MOOSE_TIMING_JSON):
        print("note: %s absent -- Moose gets no cost axis on the deviation plate"
              % MOOSE_TIMING_JSON)


def main():
    _check_inputs()
    os.makedirs(FIGURE_DIR, exist_ok=True)
    env = _env()
    print("run    %s" % CONV_JSON)
    print("moose  %s" % MOOSE_JSON)
    print("figs   %s" % FIGURE_DIR)
    for i, (cwd, name) in enumerate(PLOTTERS, 1):
        print("\n[%d/%d] %s" % (i, len(PLOTTERS) + 1, name))
        rc = subprocess.run([sys.executable, os.path.join(cwd, name)],
                            cwd=cwd, env=env).returncode
        if rc != 0:
            raise SystemExit("%s failed (exit %d)" % (name, rc))
    print("\n[%d/%d] build_plate_book.py" % (len(PLOTTERS) + 1, len(PLOTTERS) + 1))
    rc = subprocess.run(
        [sys.executable, os.path.join(PROPOSALS, "build_plate_book.py")],
        cwd=PROPOSALS, env=env).returncode
    if rc != 0:
        raise SystemExit("build_plate_book.py failed (exit %d)" % rc)
    print("\nplate book: %s" % PLATE_BOOK)


if __name__ == "__main__":
    main()
