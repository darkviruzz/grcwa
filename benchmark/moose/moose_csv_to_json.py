"""Fold a Moose sweep CSV into benchmark/moose_reference.json.

The CSV is what ``benchmark/moose/moose_convergence_bench.cs`` writes: one row
per (case, truncation order) with R/T/A and the timings. This script merges the
R values into the reference file that ``benchmark/plot_moose.py`` reads, and
parks the timings in a sibling ``moose_timing.json`` so the reference file stays
the small, hand-readable thing it is.

Sweep keys follow the reference file's convention: the Moose **max order** ``m``
for 1D cases, ``"(m,m)"`` for 2D ones (see benchmark/moose/README.md).

    python benchmark/moose/moose_csv_to_json.py <moose_conv.csv> [--dry-run]
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, os.pardir, "moose_reference.json")


def read_rows(path):
    """Successful rows of the CSV, newest-wins per (case, order)."""
    rows = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            rows[(row["case"], int(row["m_moose"]))] = row
    return rows


def sweep_key(row):
    m = int(row["m_moose"])
    return "(%d,%d)" % (m, m) if int(row["dim"]) == 2 else str(m)


def merge(reference, rows):
    """Merge into ``reference['cases']``; returns a list of change strings."""
    cases = reference.setdefault("cases", {})
    changes = []
    for (name, m), row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        entry = cases.setdefault(name, {})
        if int(row["dim"]) == 2:
            entry["dim"] = 2
        entry.setdefault("pol", "TMM" if int(row["dim"]) == 0 else row["pol"])
        sweep = entry.setdefault("sweep", {})
        key, value = sweep_key(row), float(row["R"])
        old = sweep.get(key)
        if old is None:
            changes.append("%-24s %-8s      -> %.9g" % (name, key, value))
        elif abs(old - value) > 1e-12:
            changes.append("%-24s %-8s %.9g -> %.9g" % (name, key, old, value))
        sweep[key] = value

    # "ref" is the value at the highest total order count actually present.
    for name in sorted({name for name, _ in rows}):
        entry = cases[name]
        best_key = max(entry["sweep"], key=total_orders)
        entry["ref"] = entry["sweep"][best_key]
        entry["ref_provisional"] = True
    return changes


def total_orders(key):
    """Total retained orders for a sweep key -- the same reading plot_moose.py
    uses: keys are Moose max orders m, so q = 2m+1 per axis."""
    key = key.strip()
    if key.startswith("("):
        a, b = key.strip("()").split(",")
        return (2 * int(a) + 1) * (2 * int(b) + 1)
    return 2 * int(key) + 1


def timing_block(rows):
    out = {}
    for (name, m), row in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        out.setdefault(name, {})[sweep_key(row)] = {
            "q": int(row["q"]),
            "nG": int(row["nG"]),
            "fft_refinement": int(row["fft_refinement"]),
            "t_solve_s": float(row["t_solve_s"]),
            "t_total_s": float(row["t_total_s"]),
            "mem_peak_mb": float(row["mem_peak_mb"]),
            "energy": float(row["energy"]),
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="moose_conv.csv written by the Moose script")
    parser.add_argument("--json", default=DEFAULT_JSON,
                        help="reference file to merge into (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would change, write nothing")
    args = parser.parse_args(argv)

    rows = read_rows(args.csv)
    if not rows:
        print("no successful rows in %s" % args.csv, file=sys.stderr)
        return 1

    with open(args.json) as handle:
        reference = json.load(handle)

    changes = merge(reference, rows)
    print("%d runs read, %d sweep points added or changed"
          % (len(rows), len(changes)))
    for line in changes:
        print("  " + line)

    timing_path = os.path.join(os.path.dirname(os.path.abspath(args.json)),
                               "moose_timing.json")
    if args.dry_run:
        print("dry run: %s and %s left untouched" % (args.json, timing_path))
        return 0

    with open(args.json, "w") as handle:
        json.dump(reference, handle, indent=2, sort_keys=False)
        handle.write("\n")
    with open(timing_path, "w") as handle:
        json.dump({"software": "Moose", "source": os.path.basename(args.csv),
                   "note": "wall time per (case, Moose max order); written by "
                           "benchmark/moose/moose_csv_to_json.py",
                   "cases": timing_block(rows)},
                  handle, indent=2)
        handle.write("\n")
    print("wrote %s" % args.json)
    print("wrote %s" % timing_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
