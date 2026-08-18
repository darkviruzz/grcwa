"""Material palette, geometry extraction and physics badges for the structure
figures.

The geometry is read straight out of ``benchmark/structures.py`` -- these
proposals never restate a dimension, so a figure can never drift from the
battery it claims to draw.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from structures import STRUCTURES, STRUCT, AIR, SIO2, SIN, SI, AU

# ---- material identity ------------------------------------------------------
MAT = {
    (1.0, 0.0): dict(name="air",  color="#eef4fa", edge="#9fb4c7", text="#33475b", hatch=None),
    (1.5, 0.0): dict(name="SiO2", color="#bcd8ea", edge="#5f8ba8", text="#1d3murky", hatch=None),
    (2.0, 0.0): dict(name="SiN",  color="#7fb3d3", edge="#3d6f92", text="#0f2233", hatch=None),
    (3.5, 0.0): dict(name="Si",   color="#31456b", edge="#1b2740", text="#ffffff", hatch=None),
    (0.3, 7.0): dict(name="Au",   color="#e0a92b", edge="#8a6410", text="#3a2a00", hatch="//"),
}

def mat(nk):
    key = (float(nk[0]), float(nk[1]))
    if key in MAT:
        return MAT[key]
    return dict(name=f"n={key[0]}", color="#cccccc", edge="#666", text="#000", hatch=None)

def mlabel(nk):
    m = mat(nk)
    n, k = nk
    return f"{m['name']} (n={n:g}" + (f", k={k:g})" if k else ")")

# ---- geometry ---------------------------------------------------------------
def geom(s):
    """Normalized description of a structure: layers top->bottom + pattern."""
    g = dict(name=s["name"], group=s["group"], dim=s["dim"], pol=s["pol"],
             desc=s["desc"], d=s["d"], sub=s["sub"])
    g["pol_name"] = "TM (p)" if s["pol"] == "p" else "TE (s)"
    if s["dim"] == 0:
        g.update(period=None, hi=s["film"], lo=s["film"], ff=1.0, shape="uniform")
    elif s["dim"] == 1:
        g.update(period=s["period"], hi=s["hi"], lo=s["lo"], ff=s["ff"], shape="lamellar")
    else:
        g.update(period=s["period"], hi=s["pillar"], lo=s["bg"],
                 shape=s.get("shape", "rect"))
        if g["shape"] == "circle":
            g["radius"] = s["radius"]        # in units of the period
            g["ax"] = g["ay"] = 2 * s["radius"] * s["period"]
            g["ff"] = np.pi * s["radius"] ** 2
        else:
            g["ax"], g["ay"] = s["ax"], s["ay"]
            g["ff"] = (s["ax"] * s["ay"]) / s["period"] ** 2
    return g

def orders_open(g, lam=1.0):
    """Propagating diffraction orders in the *air* half space, normal incidence."""
    if g["dim"] == 0 or g["period"] is None:
        return 1
    r = g["period"] / lam
    m = int(np.floor(r))
    if g["dim"] == 1:
        return 2 * m + 1
    n = 0
    for i in range(-m - 1, m + 2):
        for j in range(-m - 1, m + 2):
            if (i / r) ** 2 + (j / r) ** 2 < 1.0:
                n += 1
    return n

BADGE_FC = {"regime": "#eef3f8", "orders": "#eef3f8", "loss": "#fdeee2",
            "pol": "#fdecec", "contrast": "#eef3f8", "period": "#eef3f8",
            "stand": "#f2f0fa"}
BADGE_EC = {"loss": "#e0a92b", "pol": "#e8542f", "stand": "#8a5aa8"}


def badges(g, lam=1.0):
    b = []
    if g["period"]:
        r = g["period"] / lam
        b.append(("period", f"Λ/λ = {r:.2f}"))
        b.append(("regime", "sub-λ" if r < 1 else "diffracting"))
        b.append(("orders", f"{orders_open(g)} order" + ("s" if orders_open(g) != 1 else "") + " open"))
    else:
        b.append(("regime", "planar (TMM)"))
    dn = abs(complex(*g["hi"]) - complex(*g["lo"]))
    if g["dim"] > 0:
        b.append(("contrast", f"Δn = {dn:.1f}"))
    if g["hi"][1] or g["lo"][1] or g["sub"][1]:
        b.append(("loss", "lossy (metal)"))
    if (float(g["sub"][0]), float(g["sub"][1])) == (1.0, 0.0):
        b.append(("stand", "free-standing"))
    b.append(("pol", g["pol_name"]))
    return b

ALL = [geom(s) for s in STRUCTURES]
