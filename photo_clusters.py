"""End-to-end pipeline: scan a directory, update embeddings, cluster them, and write the review page.

Theory of purpose: one command takes a photo directory and a database path and leaves a review page that
reflects what is on disk now. Photos are only read.
Theory of operation: embed_photos.embed walks the directory and embeds new or changed still images. Database
rows under the directory whose files no longer exist are then removed, so deleted photos cannot be clustered.
Finally cluster_review.build writes clusters.csv and the HTML page, whose marks are tied to that clustering by a
run id. Rerunning is cheap because unchanged files are not embedded again.
"""

import argparse
import os
import sqlite3

from cluster_review import build
from embed_photos import embed


def purge_missing(root, db_path):
    """Delete rows for files under root that are gone from disk; return how many."""
    prefix = os.path.join(os.path.abspath(os.path.expanduser(root)), "")
    db = sqlite3.connect(db_path)
    gone = [p for (p,) in db.execute("SELECT path FROM images") if p.startswith(prefix) and not os.path.exists(p)]
    db.executemany("DELETE FROM images WHERE path=?", [(p,) for p in gone])
    db.commit()
    return len(gone)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root", help="directory of photos (never modified)")
    ap.add_argument("db", help="SQLite database of embeddings (created if missing)")
    ap.add_argument("--out", default="out/clusters", help="directory for clusters.csv, thumbnails and the page")
    ap.add_argument("--top", type=int, default=500, help="clusters shown on the page, largest first")
    args = ap.parse_args()
    root = os.path.abspath(os.path.expanduser(args.root))
    embed(root, args.db)
    print(f"removed {purge_missing(root, args.db)} rows for missing files", flush=True)
    run = build(root, args.db, args.out, args.top)
    print(f"run {run}: open {os.path.join(os.path.abspath(args.out), 'cluster_review.html')}")


if __name__ == "__main__":
    main()
