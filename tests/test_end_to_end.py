"""End-to-end test: sample photos -> embed -> cluster -> review page in a real browser -> apply -> re-run.

Purpose: exercise the whole pipeline the way a person uses it, through the command-line programs, the local
server and the page's own buttons, so that a change anywhere that breaks the round trip is caught.
Operation: make_sample.py writes synthetic photos into a temporary directory. photo_clusters.py embeds and
clusters them; by default with the tiny-thumbnail stand-in embedder (seconds, no PyTorch, no download), or with
the real CLIP model when E2E_MODEL=clip (downloads about 600 MB the first time). serve_review.py serves the page
on a free port, and Playwright's Chromium opens it, checks the clusters and labels, opens a full-size photo in
the lightbox, chooses "Keep best copy in all clusters", adds one per-photo exception and presses Copy JSON.
apply_review.py does a dry run of that JSON from the command line; the same plan is then executed in-process
with a temporary "trash" folder (the real one uses the macOS Trash). Finally photo_clusters.py runs again, which
must embed nothing new and find no copies left.

Requires the playwright package and its Chromium (`make test` installs both). With the stand-in embedder the
exact clusters are checked; with real CLIP only what does not depend on the model is checked (byte-identical
copies, capture-time split, and the round trip itself).
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
import apply_review as ar  # noqa: E402

CLIP = os.environ.get("E2E_MODEL", "tiny") == "clip"


def run(*args):
    out = subprocess.run([sys.executable, *args], cwd=REPO, capture_output=True, text=True)
    if out.returncode:
        raise AssertionError(f"{args[0]} failed ({out.returncode}):\n{out.stdout}\n{out.stderr}")
    return out.stdout


class EndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        t = self.tmp.name
        self.photos, self.db, self.out = f"{t}/photos", f"{t}/e.sqlite3", f"{t}/out"
        self.model_args = [] if CLIP else ["--model", "tiny-thumbnail"]

    def tearDown(self):
        self.tmp.cleanup()

    def pipeline(self):
        return run("photo_clusters.py", self.photos, self.db, "--out", self.out, "--workers", "2",
                   "--cluster-workers", "2", *self.model_args)

    def rows(self):
        with open(f"{self.out}/clusters.csv", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["rel"] = os.path.relpath(r["path"], self.photos)
        return rows

    def check_clusters(self, rows):
        by_rel = {r["rel"]: r for r in rows}
        by_id = {r["id"]: r for r in rows}
        cluster_of = lambda rel: by_rel[rel]["cluster"]
        members = lambda k: sorted(r["rel"] for r in rows if r["cluster"] == k)
        keeper_of = lambda rel: by_id[by_rel[rel]["copy_of"]]["rel"]

        # byte-identical copies: the shortest path is kept, whatever the model
        old = "backup/old laptop/trip/sunset/sunset.jpg"
        self.assertEqual(by_rel[old]["copy_kind"], "identical")
        self.assertEqual(cluster_of(old), cluster_of("trip/sunset/sunset.jpg"))
        self.assertIn(keeper_of(old), {"trip/sunset/sunset.jpg", "backup/copies/sunset.png"})
        self.assertEqual(by_rel["misc/dup/noise0.png"]["copy_kind"], "identical")
        self.assertEqual(keeper_of("misc/dup/noise0.png"), "misc/noise0.png")
        self.assertEqual(by_rel["misc/noise0.png"]["copy_of"], "")
        for r in rows:  # every copy points at a kept photo in its own cluster
            if r["copy_of"]:
                self.assertEqual(by_id[r["copy_of"]]["cluster"], r["cluster"])
                self.assertEqual(by_id[r["copy_of"]]["copy_of"], "")

        # capture time: the checkerboard taken a year later is not in the 2024 checkerboard's cluster
        if "later/checker_again0.jpg" in by_rel and "trip/checker/checker.jpg" in by_rel:
            self.assertNotEqual(cluster_of("later/checker_again0.jpg"), cluster_of("trip/checker/checker.jpg"))
        if CLIP:
            return
        self.assertEqual(members(cluster_of("misc/noise0.png")), ["misc/dup/noise0.png", "misc/noise0.png"])
        self.assertEqual(members(cluster_of("later/checker_again0.jpg")), ["later/checker_again0.jpg", "later/checker_again1.jpg"])
        for kind in ("sunset", "checker", "rings", "stripes", "blobs"):
            self.assertEqual(by_rel[f"trip/{kind}/{kind}_small.jpg"]["copy_kind"], "lower-res", kind)
            self.assertEqual(by_rel[f"backup/copies/{kind}_resaved.jpg"]["copy_kind"], "re-save", kind)
            self.assertEqual(cluster_of(f"backup/copies/{kind}.png"), cluster_of(f"trip/{kind}/{kind}.jpg"), kind)

    def review_in_browser(self, rows, exception_rel):
        from playwright.sync_api import sync_playwright

        server = subprocess.Popen([sys.executable, "serve_review.py", "--out", self.out, "--port", "0"], cwd=REPO,
                                  stdout=subprocess.PIPE, text=True)
        try:
            url = server.stdout.readline().strip()
            self.assertRegex(url, r"^http://127\.0\.0\.1:\d+/$")
            n_clusters = len({r["cluster"] for r in rows})
            n_copies = sum(bool(r["copy_of"]) for r in rows)
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(url)
                self.assertEqual(page.locator("section.cl").count(), n_clusters)
                noise = next(r for r in rows if r["rel"] == "misc/noise0.png")["cluster"]
                self.assertIn("identical files", page.locator(f"#cl{noise} h2").inner_text())
                self.assertGreater(page.locator("section.cl h2", has_text="taken 2024-06-01").count(), 0)
                self.assertIn(f"({n_copies:,} copies)", page.locator("#alldedupe").inner_text())

                # full-size photo through the server, in the lightbox
                page.locator("section.cl .im img").first.click()
                page.wait_for_function("document.getElementById('lbimg').complete && document.getElementById('lbimg').naturalWidth > 0")
                page.keyboard.press("Escape")
                self.assertFalse(page.locator("#lb").is_visible())

                page.locator("#alldedupe").click()
                if exception_rel:
                    exc = next(r for r in rows if r["rel"] == exception_rel)
                    page.locator(f'select[data-id="{exc["id"]}"]').select_option("delete")
                    self.assertIn("1 photo exceptions", page.locator("#count").inner_text())
                page.locator("#export").click()
                review = json.loads(page.locator("#json").input_value())
                browser.close()
            self.assertEqual(errors, [])
            return review
        finally:
            server.terminate()
            server.wait(10)
            server.stdout.close()

    def test_round_trip(self):
        run("make_sample.py", self.photos)
        out = self.pipeline()
        self.assertIn("embedded 33 images", out)
        rows = self.rows()
        self.check_clusters(rows)

        # a photo that is in a cluster but is no one's copy, for the per-photo exception
        exception = next((r["rel"] for r in rows if not r["copy_of"] and r["rel"].startswith("backup/copies/")
                          and not any(o["copy_of"] == r["id"] for o in rows)), None)
        review = self.review_in_browser(rows, exception)
        self.assertEqual(review["run"], ar.clusters_run(f"{self.out}/clusters.csv"))
        clusters_with_copies = {r["cluster"] for r in rows if r["copy_of"]}
        self.assertEqual({k for k, v in review["c"].items() if v == "dedupe"}, clusters_with_copies)
        self.assertEqual(set(review["x"].values()), {"delete"} if exception else set())

        # dry run from the command line
        with open(f"{self.tmp.name}/review.json", "w") as f:
            json.dump(review, f)
        n_delete = sum(bool(r["copy_of"]) for r in rows) + bool(exception)
        dry = run("apply_review.py", f"{self.tmp.name}/review.json", self.photos, self.db,
                  "--clusters", f"{self.out}/clusters.csv", "--manifest", f"{self.tmp.name}/applied.csv")
        self.assertIn("DRY RUN", dry)
        self.assertEqual(int(re.search(r"'would delete': (\d+)", dry).group(1)), n_delete)
        self.assertTrue(all(os.path.exists(r["path"]) for r in rows))

        # execute the same plan, with a temporary trash folder instead of the macOS Trash
        can = f"{self.tmp.name}/trash"
        os.makedirs(can)
        plan = ar.resolve(ar.Review(**review), rows)
        done = ar.apply_plan(plan, self.photos, self.db, f"{self.tmp.name}/docs", f"{self.tmp.name}/applied.csv",
                             True, lambda p: os.replace(p, os.path.join(can, f"{len(os.listdir(can))}-{os.path.basename(p)}")))
        self.assertEqual(done["delete"], n_delete)
        for r in rows:
            gone = bool(r["copy_of"]) or r["rel"] == exception
            self.assertEqual(os.path.exists(r["path"]), not gone, r["rel"])

        # the next run embeds nothing and finds no copies left
        again = self.pipeline()
        self.assertIn("0 to do", again)
        self.assertIn("0 identical, 0 lower-resolution and 0 re-saved copies", again)
        self.assertTrue(all(not r["copy_of"] for r in self.rows()))


if __name__ == "__main__":
    unittest.main()
