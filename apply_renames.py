"""Rename photos as approved on a rename_review.py page, with a log so every rename can be undone.

Theory of purpose: carry out reviewed renames (e.g. slide captions) safely: only approved names, nothing
overwritten, nothing silently changed.
Theory of operation: reads the plan CSV and, optionally, the page's Copy JSON (edited names and skips). The JSON's
plan id must match the plan. Names are checked (no path separators, a non-empty stem, the original extension
kept) and must be unique. A rename happens only if the source exists and the target does not. Each rename is
done with os.rename within the same folder, appended to a log CSV (old path, new path, time), and the embeddings
database row is moved to the new path. --undo LOG reverses a log. Dry run unless --execute.
"""

import argparse
import csv
import datetime
import hashlib
import json
import os
import sqlite3
import sys

from rename_review import rid


def load_plan(plan_path, review_path):
    with open(plan_path, "rb") as f:
        plan_id = hashlib.sha256(f.read()).hexdigest()[:12]
    with open(plan_path, newline="") as f:
        rows = list(csv.DictReader(f))
    review = {"plan": plan_id, "names": {}, "skip": []}
    if review_path:
        with open(review_path) as f:
            review = json.load(f)
        if review.get("plan") != plan_id:
            sys.exit(f"review is for plan {review.get('plan')}, not {plan_id}")
    moves = []
    for r in rows:
        i = rid(r["path"])
        if i in review.get("skip", []):
            continue
        name = review.get("names", {}).get(i, r["new_name"]).strip()
        moves.append((r["path"], os.path.join(os.path.dirname(r["path"]), name)))
    return moves


def check(moves):
    """Return a list of problems; an empty list means every move is safe to make."""
    problems, targets = [], {}
    for old, new in moves:
        base = os.path.basename(new)
        if "/" in base or base.startswith(".") or not os.path.splitext(base)[0].strip():
            problems.append(f"bad name: {base!r}")
        if os.path.splitext(new)[1].lower() != os.path.splitext(old)[1].lower():
            problems.append(f"extension changed: {base!r}")
        if new in targets:
            problems.append(f"two files would be named {base!r}")
        targets[new] = old
        if not os.path.exists(old):
            problems.append(f"missing: {old}")
        elif new != old and os.path.exists(new):
            problems.append(f"target exists: {new}")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("plan", nargs="?", help="plan CSV used to build the review page")
    ap.add_argument("--review", help="file holding the page's Copy JSON (default: rename as proposed)")
    ap.add_argument("--db", help="embeddings database whose paths should follow the renames")
    ap.add_argument("--log", default="out/renamed.csv")
    ap.add_argument("--undo", help="reverse the renames recorded in this log")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.undo:
        with open(args.undo, newline="") as f:
            moves = [(r["new_path"], r["old_path"]) for r in csv.DictReader(f)][::-1]
    else:
        moves = load_plan(args.plan, args.review)
    moves = [(a, b) for a, b in moves if a != b]
    problems = check(moves)
    if problems:
        print("\n".join(problems))
        sys.exit("nothing renamed")
    for a, b in moves[:5]:
        print(f"{os.path.basename(a)} -> {os.path.basename(b)}")
    if not args.execute:
        print(f"DRY RUN: {len(moves)} renames (add --execute)")
        return
    db = sqlite3.connect(args.db) if args.db else None
    new_log = not os.path.exists(args.log)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    with open(args.log, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["old_path", "new_path", "at"])
        for a, b in moves:
            os.rename(a, b)
            w.writerow([a, b, datetime.datetime.now().isoformat(timespec="seconds")])
            f.flush()
            if db:
                db.execute("UPDATE images SET path=? WHERE path=?", (b, a))
    if db:
        db.commit()
    print(f"renamed {len(moves)}")


if __name__ == "__main__":
    main()
