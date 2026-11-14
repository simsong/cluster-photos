"""Tests for apply_renames.py on real files in a temp folder: safe names, no overwrites, review edits and undo.

Purpose: renames change many files at once, so every guard must hold.
Operation: a plan CSV over generated files is checked and executed through the command-line entry point. The
tests then inspect the directory, the log and an undo, with no mocks.
"""

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, HERE)
import apply_renames as ar  # noqa: E402
from rename_review import rid  # noqa: E402


class Renames(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        for n in ("a.jpg", "b.jpg"):
            open(f"{self.d}/{n}", "w").write(n)
        self.plan = f"{self.d}/plan.csv"
        with open(self.plan, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["path", "new_name", "note", "unsure"])
            w.writerow([f"{self.d}/a.jpg", "0001 First.jpg", "", "0"])
            w.writerow([f"{self.d}/b.jpg", "0002 Second.jpg", "", "0"])

    def tearDown(self):
        self.tmp.cleanup()

    def run_cli(self, *extra):
        return subprocess.run([sys.executable, f"{HERE}/apply_renames.py", self.plan, "--log", f"{self.d}/log.csv", *extra],
                              capture_output=True, text=True)

    def test_execute_and_undo(self):
        self.assertEqual(self.run_cli("--execute").returncode, 0)
        self.assertEqual(sorted(os.listdir(self.d)), ["0001 First.jpg", "0002 Second.jpg", "log.csv", "plan.csv"])
        subprocess.run([sys.executable, f"{HERE}/apply_renames.py", "--undo", f"{self.d}/log.csv", "--log", f"{self.d}/undo.csv", "--execute"], check=True)
        self.assertTrue(os.path.exists(f"{self.d}/a.jpg") and os.path.exists(f"{self.d}/b.jpg"))

    def test_dry_run_changes_nothing(self):
        self.run_cli()
        self.assertTrue(os.path.exists(f"{self.d}/a.jpg"))

    def test_never_overwrites(self):
        open(f"{self.d}/0002 Second.jpg", "w").write("existing")
        r = self.run_cli("--execute")
        self.assertNotEqual(r.returncode, 0)
        self.assertTrue(os.path.exists(f"{self.d}/a.jpg"))  # all-or-nothing: the safe rename did not happen either
        self.assertEqual(open(f"{self.d}/0002 Second.jpg").read(), "existing")

    def test_review_edit_and_skip(self):
        rv = f"{self.d}/r.json"
        json.dump({"plan": ar.load_plan(self.plan, None) and __import__("hashlib").sha256(open(self.plan, "rb").read()).hexdigest()[:12],
                   "names": {rid(f"{self.d}/a.jpg"): "0001 Edited.jpg"}, "skip": [rid(f"{self.d}/b.jpg")]}, open(rv, "w"))
        self.assertEqual(self.run_cli("--review", rv, "--execute").returncode, 0)
        self.assertTrue(os.path.exists(f"{self.d}/0001 Edited.jpg") and os.path.exists(f"{self.d}/b.jpg"))

    def test_bad_names_rejected(self):
        self.assertIn("extension changed: '0001.png'", ar.check([(f"{self.d}/a.jpg", f"{self.d}/0001.png")]))
        self.assertTrue(any("two files" in p for p in ar.check([(f"{self.d}/a.jpg", f"{self.d}/z.jpg"), (f"{self.d}/b.jpg", f"{self.d}/z.jpg")])))


if __name__ == "__main__":
    unittest.main()
