"""Small persistent cache used by the convergence benchmark workers.

Each worker owns one cache file, so the normal benchmark does not require file
locking. Writes are atomic: an interrupted solve can lose only the point that
was in progress, while every previously completed point remains resumable.
"""
import hashlib
import json
import os
import re
import time


CACHE_SCHEMA = 1


def source_fingerprint(named_paths, extra=None):
    """Hash source files/directories plus JSON-serializable environment data."""
    digest = hashlib.sha256()
    digest.update(("conv-cache-schema:%d\n" % CACHE_SCHEMA).encode("utf-8"))
    for label, path in sorted(named_paths):
        path = os.path.abspath(path)
        if os.path.isdir(path):
            files = []
            for root, dirs, names in os.walk(path):
                dirs[:] = sorted(d for d in dirs if d != "__pycache__")
                files.extend(os.path.join(root, name) for name in sorted(names)
                             if name.endswith((".py", ".pyd", ".so", ".dll")))
            for filename in sorted(files):
                rel = os.path.relpath(filename, path).replace(os.sep, "/")
                _hash_file(digest, "%s/%s" % (label, rel), filename)
        elif os.path.isfile(path):
            _hash_file(digest, label, path)
        else:
            digest.update(("missing:%s\n" % label).encode("utf-8"))
    payload = json.dumps(extra or {}, sort_keys=True, separators=(",", ":"))
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def _hash_file(digest, label, filename):
    digest.update(("file:%s\n" % label).encode("utf-8"))
    with open(filename, "rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)


def cache_path(cache_dir, suite, module, factorization, fingerprint):
    """Return a filesystem-safe cache filename for one worker column."""
    identity = "%s-%s-%s" % (suite, module or suite, factorization)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", identity).strip("._")
    return os.path.join(cache_dir, "%s-%s.json" % (safe, fingerprint[:16]))


class PointCache:
    """JSON point cache with atomic checkpointing after every completed solve."""

    def __init__(self, path, metadata, enabled=True):
        self.path = path
        self.metadata = dict(metadata)
        self.enabled = bool(enabled)
        self.points = {}
        self.warning = None
        if self.enabled:
            self._load()

    def _load(self):
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path) as stream:
                payload = json.load(stream)
            if payload.get("schema") != CACHE_SCHEMA:
                return
            if payload.get("metadata") != self.metadata:
                return
            points = payload.get("points", {})
            if isinstance(points, dict):
                self.points = points
        except (OSError, ValueError) as exc:
            # Preserve a damaged checkpoint instead of silently overwriting it.
            backup = "%s.corrupt-%d" % (self.path, int(time.time()))
            try:
                os.replace(self.path, backup)
                self.warning = "moved unreadable cache to %s: %s" % (backup, exc)
            except OSError:
                self.enabled = False
                self.warning = "disabled unreadable cache %s: %s" % (self.path, exc)

    def get(self, key):
        value = self.points.get(key)
        return dict(value) if isinstance(value, dict) else None

    def put(self, key, value):
        if not self.enabled:
            return
        self.points[key] = dict(value)
        self._checkpoint()

    def _checkpoint(self):
        directory = os.path.dirname(self.path)
        os.makedirs(directory, exist_ok=True)
        temporary = "%s.%d.tmp" % (self.path, os.getpid())
        payload = {"schema": CACHE_SCHEMA, "metadata": self.metadata,
                   "points": self.points}
        try:
            with open(temporary, "w") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                try:
                    os.remove(temporary)
                except OSError:
                    pass
