"""Tests for embed_photos.py bookkeeping and the one-model rule in cluster_review.py.

Purpose: reruns must be incremental, a model change must re-embed, capture times must be recorded (and filled in
for databases written before the column existed), and clustering must refuse to compare vectors from different
models.
Operation: a few small generated JPEGs in a temporary directory are embedded with the tiny-thumbnail stand-in, so
no PyTorch or model download is needed. The database is then inspected and edited directly.
"""

import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cluster_review as cr  # noqa: E402
import embed_photos as ep  # noqa: E402


class Embed(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.db = f"{self.tmp.name}/photos", f"{self.tmp.name}/e.sqlite3"
        os.makedirs(self.root)
        rng = np.random.default_rng(0)
        for i in range(3):
            exif = Image.Exif()
            exif.get_ifd(0x8769)[0x9003] = f"2024:06:0{i + 1} 10:00:00"
            Image.fromarray(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8)).save(f"{self.root}/p{i}.jpg", exif=exif)
        Image.new("RGB", (30, 20), "red").save(f"{self.root}/plain.png")

    def tearDown(self):
        self.tmp.cleanup()

    def embed(self):
        ep.embed(self.root, self.db, batch=2, workers=1, model=ep.TINY)

    def rows(self):
        with sqlite3.connect(self.db) as db:
            return {os.path.basename(p): (m, t, e is not None) for p, m, t, e in
                    db.execute("SELECT path, model, taken, emb FROM images")}

    def test_records_model_and_capture_time(self):
        self.embed()
        rows = self.rows()
        self.assertEqual(rows["p1.jpg"], (ep.TINY, "2024-06-02 10:00:00", True))
        self.assertEqual(rows["plain.png"], (ep.TINY, "", True))

    def test_backfills_capture_time_without_re_embedding(self):
        self.embed()
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE images SET taken=NULL, emb=?", (b"x",))
        self.embed()
        with sqlite3.connect(self.db) as db:
            self.assertEqual({e for (e,) in db.execute("SELECT emb FROM images")}, {b"x"})  # not re-embedded
            self.assertEqual(db.execute("SELECT taken FROM images WHERE path LIKE '%p0.jpg'").fetchone()[0],
                             "2024-06-01 10:00:00")

    def test_upgrades_a_database_without_the_taken_column(self):
        with sqlite3.connect(self.db) as db:
            db.execute(ep.SCHEMA.split(";")[0].replace(",\n    taken TEXT", ""))
        ep.open_db(self.db).close()
        with sqlite3.connect(self.db) as db:
            self.assertIn("taken", {r[1] for r in db.execute("PRAGMA table_info(images)")})

    def test_model_change_re_embeds_and_mixed_models_are_refused(self):
        self.embed()
        with sqlite3.connect(self.db) as db:
            db.execute("UPDATE images SET model='other/weights' WHERE path LIKE '%p0.jpg'")
        with self.assertRaises(SystemExit) as e:
            cr.load(self.root, self.db)
        self.assertIn("other/weights (1 photos)", str(e.exception))
        self.embed()  # p0 was embedded with another model, so it is done again
        self.assertEqual({m for m, _, _ in self.rows().values()}, {ep.TINY})
        paths, E, _ = cr.load(self.root, self.db, ep.TINY)
        self.assertEqual((len(paths), E.shape[1]), (4, 768))
        with self.assertRaises(SystemExit):
            cr.load(self.root, self.db, ep.model_id(ep.MODEL, ep.PRETRAINED))


if __name__ == "__main__":
    unittest.main()
