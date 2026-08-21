"""Assemble ``plate_book.html`` from the templates plus the rendered figures.

    python benchmark/viz_proposals/make_all.py          # figures + conv_web.json
    python benchmark/viz_proposals/build_plate_book.py  # -> plate_book.html

The page is deliberately self-contained: every figure is inlined as a base64
data URI and the whole convergence payload is inlined as JSON, so the file can
be opened from disk or published as-is with no external requests.

Templates in ``plate_book/``:
  ``head.html``  <title>, the Google-Fonts link and the stylesheet
  ``body.html``  the page content, with ``{{IMG:<figure>.png}}`` placeholders
                 resolved against ``figures/``
  ``app.html``   the <script> block, with a ``{{DATA}}`` placeholder that takes
                 ``figures/conv_web.json``
"""
import base64
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "plate_book")
FIGS = os.environ.get("GRCWA_VIZ_FIGURES", os.path.join(HERE, "figures"))
OUT = os.environ.get("GRCWA_PLATE_BOOK", os.path.join(HERE, "plate_book.html"))

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml"}


def _data_uri(name):
    path = os.path.join(FIGS, name)
    if not os.path.exists(path):
        raise SystemExit("missing figure: %s\n(run make_all.py first)" % path)
    mime = MIME.get(os.path.splitext(name)[1].lower(), "application/octet-stream")
    with open(path, "rb") as f:
        return "data:%s;base64,%s" % (mime, base64.b64encode(f.read()).decode())


def _fmt(n):
    """Thousands separated with a narrow no-break space, as the sheet uses."""
    return "{:,}".format(n).replace(",", " ")


def _exp(value, fallback):
    """1e-06 is how %g spells it; 1e-6 is how the sheet spells it."""
    if not value:
        return fallback
    return ("%g" % value).replace("e-0", "e-").replace("e+0", "e")


def _meta_text(meta):
    """The {{META:*}} substitutions, all measured from the run itself.

    Nothing about the run is typed into the templates -- a plate book built from
    a half-finished sweep has to say so rather than inherit a stale boast from
    whatever run the template was written against.
    """
    q, lo, hi = meta.get("qMax"), meta.get("orderMin"), meta.get("orderMax")
    drift = meta.get("refDrift") or []
    return {
        "BUILT": meta.get("built", "?"),
        "CONV_JSON": meta.get("convJson", "?"),
        "MOOSE_JSON": meta.get("mooseJson", "?"),
        "STRUCTURES": str(meta.get("structures", "?")),
        "COLUMNS": str(meta.get("columns", "?")),
        "SOLVES": _fmt(meta["solves"]) if meta.get("solves") else "?",
        "ORDERS": ("%s … %s" % (_fmt(lo), _fmt(hi))) if lo and hi else "?",
        "QMAX": str(q) if q else "?",
        "STATE": ("complete sweep" if meta.get("complete")
                  else "sweep still running — this is the last finished stage"),
        "TOL": _exp(meta.get("tolerance"), "1e-4"),
        "LINTHRESH": _exp(meta.get("linthresh"), "1e-6"),
        "REFDRIFT": (", ".join(drift) if drift else
                     "none — every baked reference matches the current Moose file"),
    }


def build():
    with open(os.path.join(TPL, "head.html"), encoding="utf-8") as f:
        head = f.read()
    with open(os.path.join(TPL, "body.html"), encoding="utf-8") as f:
        body = f.read()
    with open(os.path.join(TPL, "app.html"), encoding="utf-8") as f:
        app = f.read()
    payload = os.path.join(FIGS, "conv_web.json")
    if not os.path.exists(payload):
        raise SystemExit("missing %s\n(run export_web.py first)" % payload)
    with open(payload, encoding="utf-8") as f:
        data = f.read()
    meta = _meta_text(json.loads(data).get("meta", {}))
    missing = sorted(set(re.findall(r"\{\{META:([A-Z_]+)\}\}", body)) - set(meta))
    if missing:
        raise SystemExit("template asks for unknown meta: %s" % ", ".join(missing))
    body = re.sub(r"\{\{META:([A-Z_]+)\}\}", lambda m: meta[m.group(1)], body)
    body = re.sub(r"\{\{IMG:([^}]+)\}\}", lambda m: _data_uri(m.group(1)), body)
    app = app.replace("{{DATA}}", data)
    return head + "\n" + body + "\n" + app + "\n"


if __name__ == "__main__":
    html = build()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote %s  (%.2f MB)" % (OUT, len(html.encode("utf-8")) / 1048576))
