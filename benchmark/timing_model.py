"""Timing aggregation and convergence summaries for the benchmark suite."""
from collections import defaultdict
import numpy as np


def split_column(column):
    """Return ``(suite, factorization)`` from ``suite[Factorization]``."""
    if "[" not in column:
        return column, "Laurent"
    suite, rule = column.rsplit("[", 1)
    return suite, rule.rstrip("]")


def isotonic_log_times(values, weights=None):
    """Weighted PAVA fit constrained to non-decreasing log-times."""
    weights = [1.0] * len(values) if weights is None else list(weights)
    blocks = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append([index, index, float(weight), float(value) * float(weight)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] <= right[3] / right[2]:
                break
            blocks[-2:] = [[left[0], right[1], left[2] + right[2],
                            left[3] + right[3]]]
    fitted = [0.0] * len(values)
    for start, end, weight, total in blocks:
        fitted[start:end + 1] = [total / weight] * (end - start + 1)
    return fitted


def build_timing_models(cases, columns):
    """Build one monotonic log-time curve per column and dimensionality.

    Measurements at a shared order count are first combined across structures
    using the median in log space. Weighted isotonic regression removes timing
    inversions without hiding BLAS regime changes; log-log interpolation between
    the fitted anchors gives the smooth plotted curve.
    Individual repeat measurements have already been reduced to their minimum
    by the worker before reaching this function.
    """
    grouped = defaultdict(list)
    for case in cases.values():
        dim = int(case.get("info", {}).get("dim", 2))
        for column in columns:
            for point in case.get("columns", {}).get(column, []):
                elapsed = point.get("time_ms")
                if elapsed is not None and elapsed > 0 and point.get("nG", 0) > 0:
                    grouped[(column, dim, int(point["nG"]))].append(float(elapsed))

    by_curve = defaultdict(list)
    for (column, dim, ng), values in grouped.items():
        representative = float(np.exp(np.median(np.log(values))))
        by_curve[(column, dim)].append({"nG": ng, "time_ms": representative,
                                        "samples": len(values)})

    models = []
    for (column, dim), samples in sorted(by_curve.items()):
        samples.sort(key=lambda sample: sample["nG"])
        fitted_logs = isotonic_log_times(
            np.log([sample["time_ms"] for sample in samples]),
            [sample["samples"] for sample in samples])
        for sample, fitted_log in zip(samples, fitted_logs):
            sample["fitted_time_ms"] = float(np.exp(fitted_log))
        suite, factorization = split_column(column)
        models.append({
            "column": column,
            "suite": suite,
            "factorization": factorization,
            "dim": dim,
            "nG_min": samples[0]["nG"],
            "nG_max": samples[-1]["nG"],
            "samples": samples,
        })
    return models


def model_lookup(models):
    return {(model["column"], int(model["dim"])): model for model in models}


def estimate_ms(model, nG):
    if not model or not nG:
        return None
    samples = model.get("samples", [])
    if not samples:
        return None
    x = np.log([sample["nG"] for sample in samples])
    y = np.log([sample["fitted_time_ms"] for sample in samples])
    target = float(np.log(nG))
    if len(samples) == 1:
        fitted = y[0]
    elif target < x[0]:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        fitted = y[0] + max(0.0, slope) * (target - x[0])
    elif target > x[-1]:
        slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
        fitted = y[-1] + max(0.0, slope) * (target - x[-1])
    else:
        fitted = np.interp(target, x, y)
    return float(np.exp(fitted))


def annotate_estimates(cases, columns, models):
    """Attach the grouped smooth timing estimate to every measured point."""
    lookup = model_lookup(models)
    for case in cases.values():
        dim = int(case.get("info", {}).get("dim", 2))
        for column in columns:
            model = lookup.get((column, dim))
            for point in case.get("columns", {}).get(column, []):
                if "R" in point:
                    point["time_est_ms"] = estimate_ms(model, point.get("nG"))


def convergence_report(cases, columns, tolerance=1e-4):
    """Find the first sustained sequence within ``tolerance``.

    At least two consecutive measured points must be in-band. A solitary
    crossing is retained as provisional, which prevents a self-reference's
    zero-error endpoint from being silently promoted to convergence.
    """
    report = []
    for case_name, case in cases.items():
        if int(case.get("info", {}).get("dim", 2)) == 0:
            continue
        reference = case.get("ref")
        if not reference or reference.get("R") is None:
            continue
        rref = float(reference["R"])
        for column in columns:
            points = sorted((point for point in case.get("columns", {}).get(column, [])
                             if "R" in point), key=lambda point: point["nG"])
            if not points:
                continue
            inside = [abs(float(point["R"]) - rref) <= tolerance
                      for point in points]
            selected = None
            status = "not_reached"
            confirmed_by = None
            for index in range(len(points) - 1):
                if inside[index] and inside[index + 1]:
                    selected = points[index]
                    confirmed_by = points[index + 1]
                    status = "converged"
                    break
            if selected is None and any(inside):
                selected = points[inside.index(True)]
                status = "provisional_crossing"
            row = {"case": case_name, "column": column,
                   "tolerance": tolerance, "status": status,
                   "reference_type": reference.get("type"),
                   "reference_provisional": bool(
                       reference.get("ref_provisional"))}
            if selected is not None:
                row.update({"nG": selected.get("nG"), "q": selected.get("q"),
                            "label": selected.get("label"),
                            "R": selected.get("R"),
                            "err_R": abs(float(selected["R"]) - rref),
                            "time_ms": selected.get("time_ms"),
                            "time_est_ms": selected.get("time_est_ms")})
            if confirmed_by is not None:
                row["confirmed_by_nG"] = confirmed_by.get("nG")
            report.append(row)
    return report
