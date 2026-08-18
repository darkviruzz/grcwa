"""Render every proposal figure into ``figures/``.

    python benchmark/viz_proposals/make_all.py
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = ["plot_s1_xsec.py", "plot_s2_iso.py", "plot_s3_atlas.py",
           "plot_c1_digitmap.py", "plot_c2_staircase.py", "plot_c3_scoreboard.py",
           "plot_c4_moose.py", "export_web.py"]

if __name__ == "__main__":
    sys.path.insert(0, HERE)
    for name in SCRIPTS:
        print("==", name)
        runpy.run_path(os.path.join(HERE, name), run_name="__main__")
