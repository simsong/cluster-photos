"""Carry out the decisions from a cluster review page: delete photos to the Trash or move them to Documents.

Theory of purpose: the page only records choices. This program acts on them, safely and reversibly.
Theory of operation: the pasted JSON is validated against a Pydantic model and resolved with clusters.csv.
A photo's own mark overrides its cluster's action, "each" or no mark means keep, and "dedupe" deletes only the
copies listed in clusters.csv (identical, lower-resolution or re-saved), as long as their keeper stays. The JSON's run id must match
clusters.csv, because cluster numbers mean nothing across runs. Unknown clusters or photos abort before any
change. The default is a dry run; --execute acts. Each file must be under --root, outside .photoslibrary, and
match its scanned sha256. Nothing is hard-deleted: deletes use the macOS Trash, moves never overwrite, and every
action is appended to a manifest and removed from the database.
"""

import argparse
import collections
import csv
import datetime
import hashlib
import os
import shutil
import sqlite3
import subprocess
import sys
from typing import Literal

from pydantic import BaseModel


class Review(BaseModel):
    """The JSON produced by the page's Copy JSON button."""

    run: str | None = None
    c: dict[str, Literal["keep", "delete", "move", "each", "dedupe"]] = {}
    x: dict[str, Literal["keep", "delete", "move"]] = {}


def clusters_run(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def resolve(review, rows):
    """Return [(cluster, id, path, action, keeper)] for photos to act on; raise ValueError on marks that match nothing."""
    clusters = {r["cluster"] for r in rows}
    ids = {r["id"] for r in rows}
    unknown = sorted(set(review.c) - clusters) + sorted(set(review.x) - ids)
    if unknown:
        raise ValueError(f"marks do not match clusters.csv: {unknown[:5]}")
    by_id = {r["id"]: r for r in rows}

    def action_for(r):
        a = review.x.get(r["id"]) or review.c.get(r["cluster"], "keep")
        if a == "dedupe":
            keeper = r.get("copy_of")
            # the keeper must itself be staying, or this copy may be the last one
            return "delete" if keeper and action_for(by_id[keeper]) == "keep" else "keep"
        return a

    plan = []
    for r in rows:
        action = action_for(r)
        if action in ("delete", "move"):
            keeper = by_id[r["copy_of"]]["path"] if r.get("copy_of") and review.x.get(r["id"]) is None \
                and review.c.get(r["cluster"]) == "dedupe" else ""
            plan.append((r["cluster"], r["id"], r["path"], action, keeper))
    return plan


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def trash(path):
    subprocess.run(["/usr/bin/trash", path], check=True, capture_output=True)
    if os.path.exists(path):
        raise RuntimeError("file is still present after trash")


def apply_plan(plan, root, db_path, dest_root, manifest, execute, trash_fn=trash):
    """Check and carry out each action; return a Counter of outcomes."""
    root = os.path.join(os.path.abspath(os.path.expanduser(root)), "")
    db = sqlite3.connect(db_path)
    out = collections.Counter()
    log = open(manifest, "a", newline="") if execute else None
    w = csv.writer(log) if log else None
    if log and log.tell() == 0:
        w.writerow(["action", "path", "dest", "size", "mtime_ns", "sha256", "cluster", "at"])
    for cluster, _, path, action, keeper in plan:
        rel = path.removeprefix(root)
        known = db.execute("SELECT sha256 FROM images WHERE path=?", (path,)).fetchone()
        if not path.startswith(root) or ".photoslibrary/" in path:
            why = "refused: outside root or inside a Photos library"
        elif not os.path.exists(path):
            why = "skipped: already gone"
        elif keeper and not os.path.exists(keeper):
            why = "skipped: its keeper is gone"
        elif not known or sha256(path) != known[0]:
            why = "skipped: changed since scan"
        else:
            why = None
        dest = os.path.join(dest_root, rel) if action == "move" else ""
        if not why and dest and os.path.exists(dest):
            why = "skipped: destination exists"
        if why:
            out[why] += 1
            print(f"{why}: {path}")
            continue
        out[f"would {action}" if not execute else action] += 1
        if not execute:
            continue
        st = os.stat(path)
        try:
            if action == "delete":
                trash_fn(path)
            else:
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.move(path, dest)
        except Exception as e:
            out[f"{action} failed"] += 1
            out[action] -= 1
            print(f"failed: {path}: {e}")
            continue
        w.writerow([action, path, dest, st.st_size, st.st_mtime_ns, known[0], cluster, datetime.datetime.now().isoformat(timespec="seconds")])
        log.flush()
        db.execute("DELETE FROM images WHERE path=?", (path,))
        db.commit()
    if log:
        log.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("review", help="file holding the JSON from Copy JSON")
    ap.add_argument("root", help="photo directory the review covers")
    ap.add_argument("db", help="SQLite database of embeddings")
    ap.add_argument("--clusters", default="out/clusters/clusters.csv")
    ap.add_argument("--dest", default="~/Documents/From Photos", help="where moved photos go, keeping subfolders")
    ap.add_argument("--manifest", default="out/applied.csv")
    ap.add_argument("--no-run-check", action="store_true", help="accept JSON from a page that has no run id")
    ap.add_argument("--execute", action="store_true", help="act; without this, only report")
    args = ap.parse_args()
    with open(args.review) as f:
        review = Review.model_validate_json(f.read())
    if not (args.no_run_check and review.run is None) and review.run != clusters_run(args.clusters):
        sys.exit(f"run id {review.run} does not match {args.clusters}; these marks belong to a different clustering")
    with open(args.clusters, newline="") as f:
        rows = list(csv.DictReader(f))
    try:
        plan = resolve(review, rows)
    except ValueError as e:
        sys.exit(str(e))
    os.makedirs(os.path.dirname(args.manifest) or ".", exist_ok=True)
    result = apply_plan(plan, args.root, args.db, os.path.expanduser(args.dest), args.manifest, args.execute)
    print(("EXECUTED: " if args.execute else "DRY RUN (add --execute to act): ") + str(dict(result)))
    if any("failed" in k for k in result):
        sys.exit(1)


if __name__ == "__main__":
    main()
