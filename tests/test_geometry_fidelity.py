"""How faithful the battery's shared masks are to its nominal geometry.

These are characterization tests: they pin down *which* structures the benchmark
actually solves, because that turned out to be the whole explanation for the 2D
disagreement with the external Moose reference (see the "The mask is not the
structure" section of benchmark/README.md).  If someone fixes
``structures.layer_mask``, KNOWN_UNFAITHFUL below goes empty and these tests
fail -- which is the intended reminder that every recorded 2D number has to be
recomputed with it.
"""
import os
import sys
import unittest

import numpy as np

BENCHMARK = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark")
if BENCHMARK not in sys.path:
    sys.path.insert(0, BENCHMARK)

import structures as ST  # noqa: E402
from geometry_fidelity import (FEATURE_TOL, exact_grid, exact_mask,  # noqa: E402
                               fidelity, report)


#: The patterned layers whose rasterized feature size misses the nominal one by
#: more than FEATURE_TOL.  All three are 2D rectangles, and all three are the
#: cases where Moose lands on a different value than grcwa and Ikarus.
KNOWN_UNFAITHFUL = {"C1_Si_pillars", "C1b_Si_pillars_diffract", "C2_Au_holes"}


class MaskFidelityTests(unittest.TestCase):

    def test_1d_masks_are_faithful(self):
        """NX_1D = 8192 renders every 1D fill factor to better than 0.01 %."""
        for s in ST.STRUCTURES:
            if s["dim"] != 1:
                continue
            f = fidelity(s)
            self.assertLess(abs(f["rel"]), 1e-4,
                            "%s: fill %.6f vs nominal %.6f"
                            % (s["name"], f["got"], f["nominal"]))

    def test_exactly_the_documented_layers_are_unfaithful(self):
        """The set of over-tolerance layers is the one the docs name."""
        bad = {f["name"] for f in (fidelity(s) for s in ST.STRUCTURES)
               if f["kind"] != "none" and abs(f["rel"]) > FEATURE_TOL}
        self.assertEqual(bad, KNOWN_UNFAITHFUL)

    def test_rect_masks_lose_a_pixel_where_documented(self):
        """The exact pixel counts the README quotes, so they cannot go stale."""
        expected = {"C1_Si_pillars": (153, 256),
                    "C1b_Si_pillars_diffract": (103, 256),
                    "C2_Au_holes": (127, 256)}
        for name, (px, n) in expected.items():
            mask, _ = ST.layer_mask(ST.STRUCT[name])
            self.assertEqual(mask.shape, (n, n))
            row = mask[:, n // 2]
            self.assertEqual(int(row.sum()), px, name)

    def test_circle_mask_is_within_tolerance(self):
        """The circle branch already samples cell centres, so D2 is fine."""
        f = fidelity(ST.STRUCT["D2_ikarus_cylinder_TE"])
        self.assertLess(abs(f["rel"]), FEATURE_TOL)


class ExactMaskTests(unittest.TestCase):

    def test_exact_grid_renders_rectangles_without_error(self):
        """An axis-aligned rectangle on a grid where w*N is an integer has no
        staircase at all -- the mask *is* the rectangle."""
        for name in ("C1_Si_pillars", "C1b_Si_pillars_diffract", "C2_Au_holes"):
            s = ST.STRUCT[name]
            n = exact_grid(s)
            mask = exact_mask(s, n)
            w_nom = s["ax"] / s["period"]
            self.assertAlmostEqual(mask[:, n // 2].sum() / n, w_nom, places=12,
                                   msg=name)
            self.assertAlmostEqual(mask.mean(), w_nom ** 2, places=12, msg=name)

    def test_exact_grid_is_at_least_as_fine_as_the_battery_grid(self):
        for name in ("C1_Si_pillars", "C1b_Si_pillars_diffract", "C2_Au_holes"):
            self.assertGreaterEqual(exact_grid(ST.STRUCT[name]), ST.NX_2D)

    def test_exact_mask_refines_the_circle(self):
        """No grid renders a circle exactly, but a finer one is closer."""
        s = ST.STRUCT["D2_ikarus_cylinder_TE"]
        coarse, _ = ST.layer_mask(s)
        fine = exact_mask(s, exact_grid(s))
        target = np.pi * s["radius"] ** 2
        self.assertLess(abs(fine.mean() - target), abs(coarse.mean() - target))

    def test_report_runs_and_returns_every_structure(self):
        rows = report()
        self.assertEqual(len(rows), len(ST.STRUCTURES))


if __name__ == "__main__":
    unittest.main()
