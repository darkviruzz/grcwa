"""Convergence worker for one solver installation and factorization mode.

Every required numerical solve is timed directly. A fast first solve may be
repeated (three total measurements by default), while expensive points run only
once. Successful points are checkpointed individually so an interrupted sweep
can resume without recalculating completed work.

Order environment variables:
  ``GRCWA_NG1D``       exact total retained orders for 1D structures;
  ``GRCWA_Q2D``        per-axis counts for 2D square blocks (total = q**2);
  ``GRCWA_NG1D_FROM_Q2D`` derive 1D orders from q plus q**2;
  ``GRCWA_MAX2D``      optional total-order safety cap (0 means no cap).

Timing/cache environment variables:
  ``GRCWA_FAST_REPEAT``       total measurements for fast points (default 3);
  ``GRCWA_FAST_THRESHOLD_MS`` repeat below this time (default 1000 ms);
  ``GRCWA_CACHE``             1 enables the persistent cache (default 0);
  ``GRCWA_CACHE_DIR``         cache directory;
  ``GRCWA_REFRESH_TIMING``    1 recomputes cached points and timings.
"""
import json
import math
import os
import platform
import sys
import time

import numpy as np

import structures as ST
from conv_cache import PointCache, cache_path, source_fingerprint


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NG1D = "1,3,5,7,9,13,17,21,25"
DEFAULT_Q2D = "1,3,5,7,9,13,17,21,25"


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def parse_odd_orders(value, name):
    """Parse a positive, odd, de-duplicated order list."""
    values = []
    seen = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        order = int(token)
        if order <= 0 or order % 2 == 0:
            raise ValueError("%s values must be positive odd integers: %s"
                             % (name, order))
        if order not in seen:
            values.append(order)
            seen.add(order)
    if not values:
        raise ValueError("%s must contain at least one order" % name)
    return values


def order_lists(environ=None):
    """Return explicit ``(1D total orders, 2D per-axis orders, 2D cap)``.

    ``GRCWA_QLIST`` remains a compatibility fallback for manual older commands;
    new jobs should use the dimension-specific variables.
    """
    environ = os.environ if environ is None else environ
    legacy = environ.get("GRCWA_QLIST")
    q2d_source = environ.get("GRCWA_Q2D", legacy or DEFAULT_Q2D)
    q2d = parse_odd_orders(q2d_source, "GRCWA_Q2D")
    derive_1d = environ.get("GRCWA_NG1D_FROM_Q2D", "").strip().lower()
    if derive_1d not in ("", "0", "false", "no", "off"):
        ng1d = sorted(set(q2d + [q * q for q in q2d]))
    elif legacy and not environ.get("GRCWA_NG1D"):
        base = parse_odd_orders(legacy, "GRCWA_QLIST")
        ng1d = []
        for q in base:
            for order in (q, q * q):
                if order not in ng1d:
                    ng1d.append(order)
    else:
        ng1d = parse_odd_orders(environ.get("GRCWA_NG1D", DEFAULT_NG1D),
                                "GRCWA_NG1D")
    max2d = int(environ.get("GRCWA_MAX2D", "0"))
    return ng1d, q2d, max2d


def planned_points(structure, ng1d, q2d, max2d=0):
    """Return ``[(q passed to solve, display label), ...]`` for a structure."""
    dim = structure["dim"]
    if dim == 0:
        return [(1, "(0D)")]
    if dim == 1:
        return [(order, str(order)) for order in ng1d]
    return [(q, "(%d,%d)" % (q, q)) for q in q2d
            if not max2d or q * q <= max2d]


def timed_solve(structure, q, label, solve, fast_repeat=3,
                fast_threshold_ms=1000.0, clock=time.perf_counter):
    """Solve once for physics and timing, repeating only a fast first solve."""
    started = clock()
    try:
        out = solve(structure, q)
    except Exception as exc:
        return {"q": q, "label": label, "error": repr(exc)}
    elapsed = clock() - started
    R, T, nG, mode = out
    if R is None:
        return {"q": q, "label": label, "skipped": mode}

    timings = [elapsed * 1e3]
    timing_error = None
    if elapsed * 1e3 < fast_threshold_ms:
        for _ in range(max(1, fast_repeat) - 1):
            started = clock()
            try:
                measured = solve(structure, q)
            except Exception as exc:
                timing_error = "repeat timing failed: %r" % (exc,)
                break
            if measured[0] is None:
                timing_error = "repeat timing skipped: %s" % measured[3]
                break
            timings.append((clock() - started) * 1e3)
    point = {"q": q, "label": label, "nG": nG, "R": R, "T": T,
             "A": 1.0 - R - T, "time_ms": min(timings),
             "time_samples_ms": timings, "timing_runs": len(timings),
             "mode": mode, "physics_cached": False, "timing_cached": False}
    if timing_error:
        point["timing_error"] = timing_error
    return point


def _package_version(distribution):
    try:
        from importlib.metadata import version
        return version(distribution)
    except Exception:
        return "?"


def configure_solver(suite, module_name, fmm):
    """Return ``(solve callable, source paths, version metadata)``."""
    if suite == "ikarus":
        import ikarus_suite as IK
        if not IK.available():
            return None, [], {"error": "ikarus not installed"}
        import ikarus

        def solve(structure, q):
            return IK.solve(structure, q, fmm)

        paths = [("structures.py", ST.__file__),
                 ("ikarus_suite.py", IK.__file__),
                 ("ikarus-package", os.path.dirname(ikarus.__file__))]
        versions = {"ikarus-rcwa": _package_version("ikarus-rcwa")}
        return solve, paths, versions

    grcwa = __import__(module_name or "grcwa")
    native = ST.supports_native_dim(grcwa)

    def solve(structure, q):
        return ST.solve(grcwa, structure, q, fmm, native)

    paths = [("structures.py", ST.__file__),
             ("grcwa-package", os.path.dirname(grcwa.__file__))]
    versions = {"grcwa-version": getattr(grcwa, "__version__", "?"),
                "native-dimensions": native}
    return solve, paths, versions


def make_cache(suite, module_name, fmm, source_paths, versions,
               fast_repeat, fast_threshold_ms):
    enabled = env_flag("GRCWA_CACHE", False)
    physics_identity = {
        "suite": suite,
        "module": module_name,
        "factorization": fmm,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "versions": versions,
        "cache_tag": os.environ.get("GRCWA_CACHE_TAG", ""),
    }
    fingerprint = source_fingerprint(source_paths, physics_identity)
    directory = os.environ.get("GRCWA_CACHE_DIR",
                               os.path.join(HERE, ".cache", "convergence"))
    path = cache_path(directory, suite, module_name, str(fmm), fingerprint)
    metadata = {"fingerprint": fingerprint, "identity": physics_identity}
    timing_identity = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": {name: os.environ.get(name) for name in
                    ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                     "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }
    timing_fingerprint = source_fingerprint([], timing_identity)
    return (PointCache(path, metadata, enabled=enabled), timing_fingerprint,
            timing_identity)


def _valid_samples(values):
    return [float(value) for value in values or []
            if isinstance(value, (int, float)) and math.isfinite(value)
            and value > 0]


def _result_from_record(record, samples, physics_cached, timing_cached,
                        timing_error=None, timing_stale=False):
    point = dict(record["result"])
    point.update({"time_ms": min(samples) if samples else None,
                  "time_samples_ms": samples,
                  "timing_runs": len(samples),
                  "physics_cached": physics_cached,
                  "timing_cached": timing_cached})
    if timing_error:
        point["timing_error"] = timing_error
    if timing_stale:
        point["timing_stale"] = True
    return point


def cached_timed_solve(structure, q, label, solve, cache, key,
                       timing_fingerprint, timing_identity, fast_repeat,
                       fast_threshold_ms, refresh, stats,
                       clock=time.perf_counter):
    """Return a point while independently resuming physics and timing data."""
    record = cache.get(key) if cache.enabled else None
    result = record.get("result") if isinstance(record, dict) else None
    timing = record.get("timing", {}) if isinstance(record, dict) else {}
    same_environment = timing.get("fingerprint") == timing_fingerprint
    fallback_samples = _valid_samples(timing.get("samples_ms"))
    samples = list(fallback_samples) if same_environment else []
    if refresh:
        samples = []

    if result is None:
        point = timed_solve(structure, q, label, solve, fast_repeat,
                            fast_threshold_ms, clock)
        if "R" not in point:
            stats["unsolved"] += 1
            return point
        stats["physics_solved"] += 1
        result = {name: point[name] for name in
                  ("q", "label", "nG", "R", "T", "A", "mode")}
        samples = list(point["time_samples_ms"])
        record = {"result": result,
                  "timing": {"fingerprint": timing_fingerprint,
                             "identity": timing_identity,
                             "samples_ms": samples}}
        cache.put(key, record)
        stats["timing_samples"] += len(samples)
        if point.get("timing_error"):
            stats["timing_failures"] += 1
        return _result_from_record(record, samples, False, False,
                                   point.get("timing_error"))

    physics_cached = True
    required = (fast_repeat if samples and samples[0] < fast_threshold_ms else 1)
    timing_cached = bool(samples) and len(samples) >= required and not refresh
    if not samples:
        required = 1

    timing_error = None
    while len(samples) < required:
        started = clock()
        try:
            measured = solve(structure, q)
        except Exception as exc:
            timing_error = "timing solve failed: %r" % (exc,)
            break
        elapsed_ms = (clock() - started) * 1e3
        if measured[0] is None:
            timing_error = "timing solve skipped: %s" % measured[3]
            break
        samples.append(elapsed_ms)
        if len(samples) == 1 and elapsed_ms < fast_threshold_ms:
            required = fast_repeat
        record = {"result": result,
                  "timing": {"fingerprint": timing_fingerprint,
                             "identity": timing_identity,
                             "samples_ms": samples}}
        cache.put(key, record)
        stats["timing_samples"] += 1

    if timing_error:
        stats["timing_failures"] += 1
        usable = samples or fallback_samples
        return _result_from_record(
            {"result": result}, usable, physics_cached, False,
            timing_error=timing_error,
            timing_stale=not samples and bool(fallback_samples))
    if not samples:
        stats["timing_failures"] += 1
        return _result_from_record(
            {"result": result}, fallback_samples, physics_cached, False,
            timing_error="no timing sample",
            timing_stale=bool(fallback_samples))
    if timing_cached:
        stats["full_hits"] += 1
    else:
        stats["timing_resumed"] += 1
    return _result_from_record(record, samples, physics_cached, timing_cached)


def run_structure(structure, solve, cache, timing_fingerprint, timing_identity,
                   ng1d, q2d, max2d, fast_repeat, fast_threshold_ms, refresh,
                  stats, progress=None):
    sweep = []
    plan = planned_points(structure, ng1d, q2d, max2d)
    for index, (q, label) in enumerate(plan, 1):
        if progress:
            progress(structure, q, label, index, len(plan), None)
        key = "%s|dim=%s|q=%s" % (structure["name"], structure["dim"], q)
        point = cached_timed_solve(
            structure, q, label, solve, cache, key, timing_fingerprint,
            timing_identity, fast_repeat, fast_threshold_ms, refresh, stats)
        sweep.append(point)
        if progress:
            progress(structure, q, label, index, len(plan), point)

    info = {key: structure[key] for key in ("group", "dim", "pol", "desc")}
    info["nk"] = {key: list(structure[key]) for key in
                  ("hi", "lo", "film", "sub", "pillar", "bg")
                  if key in structure}
    for key in ("period", "ff", "d", "ax", "ay", "radius"):
        if key in structure:
            info[key] = structure[key]
    if "shape" in structure:
        info["shape"] = structure["shape"]
    kept = [point for point in sweep
            if "R" in point or "skipped" in point or "error" in point]
    return {"info": info, "sweep": kept}


def main():
    suite = os.environ.get("SUITE", "grcwa")
    raw_fmm = os.environ.get("FMM", "none")
    fmm = None if raw_fmm == "none" else raw_fmm
    module_name = os.environ.get("GRCWA_MOD", "grcwa")
    fast_repeat = max(1, int(os.environ.get("GRCWA_FAST_REPEAT", "3")))
    fast_threshold_ms = max(
        0.0, float(os.environ.get("GRCWA_FAST_THRESHOLD_MS", "1000")))
    refresh = env_flag("GRCWA_REFRESH_TIMING", False)
    ng1d, q2d, max2d = order_lists()

    solve, source_paths, versions = configure_solver(suite, module_name, fmm)
    if solve is None:
        print(json.dumps({"_error": versions["error"]}))
        return
    cache, timing_fingerprint, timing_identity = make_cache(
        suite, module_name, fmm, source_paths, versions, fast_repeat,
        fast_threshold_ms)
    if cache.warning:
        print("cache warning: " + cache.warning, file=sys.stderr, flush=True)

    stats = {"full_hits": 0, "physics_solved": 0, "timing_resumed": 0,
             "timing_samples": 0, "timing_failures": 0, "unsolved": 0}

    def report_progress(structure, q, label, index, total, point):
        prefix = "      point %d/%d q=%s" % (index, total, label)
        if point is None:
            print(prefix + " starting", file=sys.stderr, flush=True)
            return
        if "error" in point:
            outcome = "ERROR " + point["error"]
        elif "skipped" in point:
            outcome = "SKIPPED " + str(point["skipped"])
        else:
            source = "cache" if point.get("physics_cached") else "solved"
            elapsed = point.get("time_ms")
            timing = "no timing" if elapsed is None else "%.3f ms" % elapsed
            outcome = "%s nG=%s, %s" % (source, point.get("nG"), timing)
            if point.get("timing_error"):
                outcome += ", TIMING ERROR " + point["timing_error"]
        print(prefix + " -> " + outcome, file=sys.stderr, flush=True)

    results = {}
    total_structures = len(ST.STRUCTURES)
    for structure_index, structure in enumerate(ST.STRUCTURES, 1):
        print("structure %d/%d: %s" % (
            structure_index, total_structures, structure["name"]),
            file=sys.stderr, flush=True)
        results[structure["name"]] = run_structure(
            structure, solve, cache, timing_fingerprint, timing_identity, ng1d,
            q2d, max2d, fast_repeat, fast_threshold_ms, refresh, stats,
            progress=report_progress)
    print("cache: %d full hit(s), %d new physics point(s), "
          "%d timing resume(s), %d new timing sample(s), "
          "%d timing failure(s), %d skipped/error"
          % (stats["full_hits"], stats["physics_solved"],
              stats["timing_resumed"], stats["timing_samples"],
              stats["timing_failures"], stats["unsolved"]),
          file=sys.stderr, flush=True)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
