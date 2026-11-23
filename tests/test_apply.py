"""Tests for apply_review.py using real files, a real SQLite database and real moves in a temp directory.

Purpose: decisions must resolve as the page promises, and safety checks must stop unsafe actions.
Operation: photos are small byte files whose sha256 is stored in a temporary images table. The "trash" is a
function that moves a file into a temp folder, so no test touches the real Trash. Each test checks files on
disk afterward, as well as the returned outcome counts and the manifest.
"""

import hashlib
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import apply_review as ar  # noqa: E402


class Apply(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.root, self.can, self.dest = f"{t}/photos", f"{t}/can", f"{t}/docs"
        os.makedirs(f"{self.root}/a")
        os.makedirs(self.can)
        self.db = f"{t}/e.sqlite3"
        self.man = f"{t}/m.csv"
        db = sqlite3.connect(self.db)
        db.execute("CREATE TABLE images (path TEXT PRIMARY KEY, sha256 TEXT)")
        for n in ("one", "two", "three"):
            p = self.p(n)
            open(p, "wb").write(n.encode())
            db.execute("INSERT INTO images VALUES (?,?)", (p, hashlib.sha256(n.encode()).hexdigest()))
        db.commit()
        db.close()
        self.rows = [{"cluster": "1", "id": n, "path": self.p(n)} for n in ("one", "two")] + \
                    [{"cluster": "2", "id": "three", "path": self.p("three")}]

    def tearDown(self):
        self.tmp.cleanup()

    def p(self, n):
        return f"{self.root}/a/{n}.jpg"

    def fake_trash(self, path):
        os.replace(path, os.path.join(self.can, os.path.basename(path)))

    def run_plan(self, review, execute=True):
        plan = ar.resolve(ar.Review(**review), self.rows)
        return ar.apply_plan(plan, self.root, self.db, self.dest, self.man, execute, self.fake_trash)

    def test_exception_overrides_cluster_and_each_means_keep(self):
        plan = ar.resolve(ar.Review(c={"1": "delete", "2": "each"}, x={"two": "keep"}), self.rows)
        self.assertEqual([(p[1], p[3]) for p in plan], [("one", "delete")])

    def dedupe_rows(self):
        rows = [dict(r, copy_of="") for r in self.rows]
        rows[1]["copy_of"] = "one"
        return rows

    def test_dedupe_deletes_only_listed_copies(self):
        plan = ar.resolve(ar.Review(c={"1": "dedupe"}), self.dedupe_rows())
        self.assertEqual([(p[1], p[3], p[4]) for p in plan], [("two", "delete", self.p("one"))])

    def test_dedupe_keeps_copy_when_keeper_is_being_removed(self):
        self.assertEqual(ar.resolve(ar.Review(c={"1": "dedupe"}, x={"one": "delete"}), self.dedupe_rows())[0][1], "one")
        self.assertEqual(len(ar.resolve(ar.Review(c={"1": "dedupe"}, x={"one": "delete"}), self.dedupe_rows())), 1)

    def test_dedupe_skips_when_keeper_is_gone(self):
        plan = ar.resolve(ar.Review(c={"1": "dedupe"}), self.dedupe_rows())
        os.remove(self.p("one"))
        out = ar.apply_plan(plan, self.root, self.db, self.dest, self.man, True, self.fake_trash)
        self.assertEqual(out["skipped: its keeper is gone"], 1)
        self.assertTrue(os.path.exists(self.p("two")))

    def test_unknown_mark_is_rejected(self):
        with self.assertRaises(ValueError):
            ar.resolve(ar.Review(c={"99": "delete"}), self.rows)

    def test_dry_run_changes_nothing(self):
        out = self.run_plan({"c": {"1": "delete"}}, execute=False)
        self.assertEqual(out["would delete"], 2)
        self.assertTrue(os.path.exists(self.p("one")))
        self.assertFalse(os.path.exists(self.man))

    def test_delete_and_move(self):
        out = self.run_plan({"c": {"1": "delete"}, "x": {"three": "move"}})
        self.assertEqual((out["delete"], out["move"]), (2, 1))
        self.assertTrue(os.path.exists(f"{self.can}/one.jpg"))
        self.assertTrue(os.path.exists(f"{self.dest}/a/three.jpg"))
        self.assertFalse(os.path.exists(self.p("three")))
        self.assertEqual(sum(1 for _ in open(self.man)), 4)

    def test_changed_file_is_skipped(self):
        open(self.p("one"), "wb").write(b"edited")
        out = self.run_plan({"c": {"1": "delete"}})
        self.assertEqual(out["skipped: changed since scan"], 1)
        self.assertTrue(os.path.exists(self.p("one")))

    def test_move_never_overwrites(self):
        os.makedirs(f"{self.dest}/a")
        open(f"{self.dest}/a/three.jpg", "wb").write(b"existing")
        out = self.run_plan({"c": {"2": "move"}})
        self.assertEqual(out["skipped: destination exists"], 1)
        self.assertEqual(open(f"{self.dest}/a/three.jpg", "rb").read(), b"existing")
        self.assertTrue(os.path.exists(self.p("three")))

    def test_path_outside_root_is_refused(self):
        plan = [("1", "x", f"{self.tmp.name}/elsewhere.jpg", "delete", "")]
        out = ar.apply_plan(plan, self.root, self.db, self.dest, self.man, True, self.fake_trash)
        self.assertEqual(out["refused: outside root or inside a Photos library"], 1)


if __name__ == "__main__":
    unittest.main()
