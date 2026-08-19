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


def build():
    with open(os.path.join(TPL, "head.html")) as f:
        head = f.read()
    with open(os.path.join(TPL, "body.html")) as f:
        body = f.read()
    with open(os.path.join(TPL, "app.html")) as f:
        app = f.read()
    body = re.sub(r"\{\{IMG:([^}]+)\}\}", lambda m: _data_uri(m.group(1)), body)
    payload = os.path.join(FIGS, "conv_web.json")
    if not os.path.exists(payload):
        raise SystemExit("missing %s\n(run export_web.py first)" % payload)
    with open(payload) as f:
        app = app.replace("{{DATA}}", f.read())
    return head + "\n" + body + "\n" + app + "\n"


if __name__ == "__main__":
    html = build()
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print("wrote %s  (%.2f MB)" % (OUT, len(html.encode("utf-8")) / 1048576))
