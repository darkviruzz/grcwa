"""Render every figure -- the shipped plotters and the proposals -- into
``figures/``, which is the committed gallery for this branch.

    python benchmark/viz_proposals/make_all.py
"""
import os
import runpy
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
FIGS = os.path.join(HERE, "figures")

# adopted -- these live in benchmark/ and are part of the normal plotting run
SHIPPED = ["plot_structures.py", "plot_conv_cost.py", "plot_deviation.py"]
# still proposals -- S1 (cross-sections, to be reworked), C1 (digit map),
# C3 (scoreboard), and the JSON behind the interactive plate book
PROPOSED = ["plot_s1_xsec.py", "plot_c1_digitmap.py", "plot_c3_scoreboard.py",
            "export_web.py"]

if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    env = dict(os.environ, GRCWA_PLOT_OUTPUT_DIR=FIGS)
    for name in SHIPPED:
        print("==", name)
        subprocess.run([sys.executable, os.path.join(BENCH, name)], check=True,
                       cwd=BENCH, env=env)
    sys.path.insert(0, HERE)
    for name in PROPOSED:
        print("==", name)
        runpy.run_path(os.path.join(HERE, name), run_name="__main__")
