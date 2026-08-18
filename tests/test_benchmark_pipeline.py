"""Focused tests for the resumable convergence benchmark infrastructure."""
import os
import re
import sys
import tempfile
import unittest


BENCHMARK = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmark")
if BENCHMARK not in sys.path:
    sys.path.insert(0, BENCHMARK)

from conv_cache import PointCache  # noqa: E402
from conv_worker import (cached_timed_solve, order_lists, planned_points,
                         timed_solve)  # noqa: E402
from conv_run import validate_worker_results  # noqa: E402
from timing_model import (build_timing_models, convergence_report,
                          estimate_ms)  # noqa: E402


def fake_clock(values):
    iterator = iter(values)
    return lambda: next(iterator)


def result(R=0.25, nG=5):
    return R, 1.0 - R, nG, "native"


def empty_stats():
    return {"full_hits": 0, "physics_solved": 0, "timing_resumed": 0,
            "timing_samples": 0, "timing_failures": 0, "unsolved": 0}


class BenchmarkPipelineTests(unittest.TestCase):

    def test_split_order_plans_include_2d_q61(self):
        env = {"GRCWA_NG1D": "1,5,9,61",
               "GRCWA_Q2D": "1,3,5,61", "GRCWA_MAX2D": "3721"}
        ng1d, q2d, cap = order_lists(env)
        self.assertEqual(planned_points({"dim": 0}, ng1d, q2d, cap),
                         [(1, "(0D)")])
        self.assertEqual([q for q, _ in planned_points(
            {"dim": 1}, ng1d, q2d, cap)], [1, 5, 9, 61])
        self.assertEqual([q * q for q, _ in planned_points(
            {"dim": 2}, ng1d, q2d, cap)], [1, 9, 25, 3721])
        self.assertEqual([q for q, _ in planned_points(
            {"dim": 2}, ng1d, q2d, 3720)], [1, 3, 5])
        derived, derived_q2d, _ = order_lists({
            "GRCWA_Q2D": "1,3,5,15", "GRCWA_NG1D_FROM_Q2D": "1"})
        self.assertEqual(derived_q2d, [1, 3, 5, 15])
        self.assertEqual(derived, [1, 3, 5, 9, 15, 25, 225])
        with self.assertRaises(ValueError):
            order_lists({"GRCWA_NG1D": "1,4", "GRCWA_Q2D": "1,3"})

    def test_result_solve_is_timed_and_only_fast_points_repeat(self):
        calls = []

        def solve(structure, q):
            calls.append(q)
            return result(nG=q)

        slow = timed_solve({}, 5, "5", solve, fast_repeat=3,
                           fast_threshold_ms=100,
                           clock=fake_clock([0.0, 0.2]))
        self.assertEqual(calls, [5])
        self.assertEqual(slow["timing_runs"], 1)
        self.assertAlmostEqual(slow["time_ms"], 200.0)

        calls[:] = []
        fast = timed_solve(
            {}, 5, "5", solve, fast_repeat=3, fast_threshold_ms=100,
            clock=fake_clock([0.0, 0.05, 1.0, 1.04, 2.0, 2.01]))
        self.assertEqual(calls, [5, 5, 5])
        self.assertEqual(fast["timing_runs"], 3)
        for actual, expected in zip(fast["time_samples_ms"], [50, 40, 10]):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(fast["time_ms"], 10.0)
        self.assertEqual(fast["R"], 0.25)

        calls[:] = []
        subsecond = timed_solve(
            {}, 5, "5", solve,
            clock=fake_clock([0.0, 0.8, 1.0, 1.7, 2.0, 2.6]))
        self.assertEqual(calls, [5, 5, 5])
        self.assertEqual(subsecond["timing_runs"], 3)
        self.assertAlmostEqual(subsecond["time_ms"], 600.0)

    def test_cache_reuses_physics_timing_and_extends_orders(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "points.json")
            metadata = {"fingerprint": "physics-v1"}
            cache = PointCache(path, metadata, enabled=True)
            calls = []

            def solve(structure, q):
                calls.append(q)
                return result(R=0.2 + q / 100.0, nG=q)

            point = cached_timed_solve(
                {"name": "case", "dim": 1}, 5, "5", solve, cache,
                "case|q=5", "timing-v1", {"machine": "test"}, 3, 100,
                False, empty_stats(), clock=fake_clock([0.0, 0.2]))
            self.assertFalse(point["physics_cached"])
            self.assertEqual(calls, [5])

            reloaded = PointCache(path, metadata, enabled=True)
            calls[:] = []
            hit = cached_timed_solve(
                {"name": "case", "dim": 1}, 5, "5", solve, reloaded,
                "case|q=5", "timing-v1", {"machine": "test"}, 3, 100,
                False, empty_stats())
            self.assertEqual(calls, [])
            self.assertTrue(hit["physics_cached"])
            self.assertTrue(hit["timing_cached"])
            self.assertEqual(hit["R"], point["R"])

            added = cached_timed_solve(
                {"name": "case", "dim": 1}, 7, "7", solve, reloaded,
                "case|q=7", "timing-v1", {"machine": "test"}, 3, 100,
                False, empty_stats(), clock=fake_clock([0.0, 0.2]))
            self.assertEqual(calls, [7])
            self.assertFalse(added["physics_cached"])

    def test_timing_refresh_preserves_cached_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = PointCache(os.path.join(directory, "points.json"),
                               {"id": 1}, enabled=True)
            key = "case|q=5"
            cache.put(key, {
                "result": {"q": 5, "label": "5", "nG": 5, "R": 0.25,
                           "T": 0.75, "A": 0.0, "mode": "native"},
                "timing": {"fingerprint": "old", "identity": {},
                           "samples_ms": [10.0]},
            })
            calls = []

            def solve(structure, q):
                calls.append(q)
                return result(R=0.99, nG=q)

            refreshed = cached_timed_solve(
                {"name": "case", "dim": 1}, 5, "5", solve, cache, key,
                "new", {"machine": "test"}, 3, 100, True, empty_stats(),
                clock=fake_clock([0.0, 0.2]))
            self.assertEqual(calls, [5])
            self.assertEqual(refreshed["R"], 0.25)
            self.assertTrue(refreshed["physics_cached"])
            self.assertFalse(refreshed["timing_cached"])

            def failed_solve(structure, q):
                raise RuntimeError("temporary backend failure")

            stats = empty_stats()
            failed_refresh = cached_timed_solve(
                {"name": "case", "dim": 1}, 5, "5", failed_solve, cache,
                key, "newer", {"machine": "test"}, 3, 100, True, stats,
                clock=fake_clock([0.0]))
            self.assertEqual(failed_refresh["R"], 0.25)
            self.assertEqual(failed_refresh["time_ms"], 200.0)
            self.assertTrue(failed_refresh["timing_stale"])
            self.assertIn("temporary backend failure",
                          failed_refresh["timing_error"])
            self.assertEqual(stats["timing_failures"], 1)

    def test_completeness_validation_rejects_failed_high_order(self):
        structure = {"name": "hard", "dim": 2}
        good = {"q": 1, "nG": 1, "R": 0.2, "T": 0.8, "A": 0.0,
                "time_ms": 1.0, "time_samples_ms": [1.0]}
        failed = {"q": 41, "error": "out of memory"}
        attempts = {"fork[Pol]": {
            "hard": {"info": {"dim": 2}, "sweep": [good, failed]}}}
        issues = validate_worker_results(
            attempts, ["fork[Pol]"], [structure], [1], [1, 41], 1681,
            {"fork[Pol]"}, fast_repeat=1)
        self.assertTrue(any("q=41" in issue and "out of memory" in issue
                            for issue in issues))

        attempts["fork[Pol]"]["hard"]["sweep"][1] = dict(
            good, q=41, nG=1681)
        self.assertEqual(validate_worker_results(
            attempts, ["fork[Pol]"], [structure], [1], [1, 41], 1681,
            {"fork[Pol]"}, fast_repeat=1), [])

    def test_timing_models_are_grouped_and_monotonic(self):
        cases = {
            "a": {"info": {"dim": 1}, "columns": {
                "fork[Pol]": [{"nG": 1, "R": 0, "time_ms": 10},
                               {"nG": 9, "R": 0, "time_ms": 8}],
                "ikarus[NV]": [{"nG": 1, "R": 0, "time_ms": 2},
                                {"nG": 9, "R": 0, "time_ms": 20}]}},
            "b": {"info": {"dim": 2}, "columns": {
                "fork[Pol]": [{"nG": 1, "R": 0, "time_ms": 100},
                               {"nG": 9, "R": 0, "time_ms": 900}]}}
        }
        models = build_timing_models(cases, ["fork[Pol]", "ikarus[NV]"])
        self.assertEqual({(model["column"], model["dim"]) for model in models},
                         {("fork[Pol]", 1), ("fork[Pol]", 2),
                          ("ikarus[NV]", 1)})
        for model in models:
            fitted = [sample["fitted_time_ms"]
                      for sample in model["samples"]]
            self.assertEqual(fitted, sorted(fitted))
            self.assertGreater(estimate_ms(model, model["nG_min"]), 0)

    def test_convergence_requires_sustained_two_point_sequence(self):
        values = [2e-4, 8e-5, 2e-4, 7e-5, 6e-5]
        points = [{"nG": index + 1, "q": index + 1,
                   "label": str(index + 1), "R": value, "time_ms": 1.0,
                   "time_est_ms": 1.0}
                  for index, value in enumerate(values)]
        cases = {"hard": {"info": {"dim": 1}, "ref": {
                 "R": 0.0, "type": "external_moose"},
                 "columns": {"fork[Pol]": points}}}
        report = convergence_report(cases, ["fork[Pol]"], 1e-4)[0]
        self.assertEqual(report["status"], "converged")
        self.assertEqual(report["nG"], 4)
        self.assertEqual(report["confirmed_by_nG"], 5)

        cases["hard"]["columns"]["fork[Pol]"] = points[:4]
        provisional = convergence_report(cases, ["fork[Pol]"], 1e-4)[0]
        self.assertEqual(provisional["status"], "provisional_crossing")

        cases["anchor"] = {"info": {"dim": 0}, "ref": {
            "R": 0.0, "type": "analytic_exact"},
            "columns": {"fork[Pol]": [points[0]]}}
        self.assertEqual(len(convergence_report(cases, ["fork[Pol]"], 1e-4)), 1)

    def test_night_batch_is_incremental_and_reaches_q61(self):
        with open(os.path.join(BENCHMARK, "run_overnight.bat")) as stream:
            text = stream.read().lower()
        self.assertNotIn('"benchmark\\run.py"', text)
        self.assertNotIn('"benchmark\\plot_benchmark.py"', text)
        initial = re.search(r'set "initial_q_list=([0-9,]+)"', text)
        full = re.search(r'set "full_q_list=([0-9,]+)"', text)
        growth = re.search(r'set "grow_orders=([0-9 ]+)"', text)
        self.assertIsNotNone(initial)
        self.assertIsNotNone(full)
        self.assertIsNotNone(growth)
        self.assertEqual(initial.group(1), "1,3,5,7,9,11,13,15")
        self.assertEqual([int(value) for value in full.group(1).split(",")],
                         list(range(1, 62, 2)))
        self.assertEqual([int(value) for value in growth.group(1).split()],
                         list(range(17, 62, 2)))
        self.assertIn('set "stage_total=24"', text)
        self.assertIn("grcwa_max2d=3721", text)
        self.assertIn("grcwa_fast_threshold_ms=1000", text)
        self.assertIn('call :run_snapshot "%initial_q_max%"', text)
        self.assertIn('set "grcwa_ng1d_from_q2d=1"', text)
        self.assertIn('grcwa_ng1d=sorted union of q and q^^2', text)
        self.assertIn('"benchmark\\plot_conv.py"', text)
        self.assertIn("grcwa_conv_tol=1e-4", text)
        self.assertIn("grcwa_cache=1", text)
        self.assertIn("grcwa_required_columns=", text)
        self.assertIn("grcwa_output_dir=%cd%\\benchmark", text)
        self.assertIn("usage: %~nx0", text)


if __name__ == "__main__":
    unittest.main()
