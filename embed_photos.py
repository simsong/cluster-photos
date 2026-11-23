"""Compute CLIP image embeddings for every still image under a root and store them in sqlite.

Theory of purpose: give the clustering step a perceptual signal ("these two photos look
the same") that survives re-encoding, resizing and metadata changes.
Theory of operation: walk the root (read-only), skip files whose (path, size, mtime_ns, model)
row is already present, and feed the rest through a DataLoader whose workers read each
file once to compute sha256, read the EXIF capture time, and decode it (EXIF-transposed, HEIC via pillow-heif).
The main process runs open_clip on CUDA, MPS or the CPU and stores L2-normalized float32 vectors as raw
blobs. Changing --model or --pretrained re-embeds every file, because vectors from different models cannot be
compared. Rows from before the capture-time column existed get their time filled in without re-embedding.
--model tiny-thumbnail is a stand-in for tests: a 16 x 16 colour thumbnail as the vector, with no PyTorch and
no model download. It finds copies of one picture, but not semantically similar scenes.
"""

import argparse
import hashlib
import io
import os
import sqlite3
import sys
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageOps

from photo_io import capture_time, capture_time_of, register_heif

IMAGE_EXTS = {".jpg", ".jpeg", ".heic", ".png", ".gif", ".tif", ".tiff", ".webp", ".bmp"}
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"
TINY = "tiny-thumbnail"

SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    path TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    sha256 TEXT,
    width INTEGER,
    height INTEGER,
    model TEXT,
    emb BLOB,
    error TEXT,
    taken TEXT
);
CREATE INDEX IF NOT EXISTS images_sha ON images(sha256);
"""
COLUMNS = ("path", "size", "mtime_ns", "sha256", "width", "height", "model", "emb", "error", "taken")


def model_id(model, pretrained):
    """The string stored in images.model; vectors with different ids are never compared."""
    return TINY if model == TINY else f"{model}/{pretrained}"


def open_db(db_path):
    """Open (creating or upgrading) the embeddings database."""
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    if "taken" not in {r[1] for r in db.execute("PRAGMA table_info(images)")}:
        db.execute("ALTER TABLE images ADD COLUMN taken TEXT")  # NULL = not read yet, '' = no capture time
    return db


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if not name.startswith(".") and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, name)


def read_photo(path, size, mtime_ns, draft=448):
    """Read a file once: return (metadata dict, upright RGB image or None on error)."""
    register_heif()
    meta = {"path": path, "size": size, "mtime_ns": mtime_ns, "sha256": "", "width": 0, "height": 0, "error": "",
            "taken": ""}
    try:
        with open(path, "rb") as f:
            data = f.read()
        meta["sha256"] = hashlib.sha256(data).hexdigest()
        img = Image.open(io.BytesIO(data))
        w, h = img.size  # true pixel size, before draft() downscales the decode
        meta["width"], meta["height"] = (h, w) if img.getexif().get(0x0112) in (5, 6, 7, 8) else (w, h)
        meta["taken"] = capture_time(img)
        img.draft("RGB", (draft, draft))  # fast JPEG DCT downscale; no-op for other formats
        return meta, ImageOps.exif_transpose(img).convert("RGB")
    except Exception as e:  # corrupt or unsupported file: record it and move on
        meta["error"] = f"{type(e).__name__}: {e}"[:500]
        return meta, None


class Photos:
    """Map-style dataset for torch's DataLoader (any object with __len__ and __getitem__ will do)."""

    def __init__(self, items, preprocess):
        self.items, self.preprocess = items, preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch
        meta, img = read_photo(*self.items[i])
        return (torch.zeros(3, 224, 224) if img is None else self.preprocess(img)), meta


def collate(batch):
    import torch
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def tiny_vector(img):
    """The stand-in embedding: a mean-centred, L2-normalized 16 x 16 RGB thumbnail (768 numbers)."""
    v = np.asarray(img.resize((16, 16), Image.Resampling.BOX), np.float32).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def _tiny(item):
    meta, img = read_photo(*item, draft=64)
    return meta, (None if img is None else tiny_vector(img))


def batches_tiny(todo, batch, workers):
    with Pool(workers) as pool:
        out = pool.imap(_tiny, todo, chunksize=8)
        while chunk := [x for _, x in zip(range(batch), out)]:
            yield [m for m, _ in chunk], [e for _, e in chunk]


def batches_clip(todo, batch, workers, model, pretrained):
    import open_clip
    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    net, _, preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
    net.eval()
    loader = DataLoader(Photos(todo, preprocess), batch_size=batch, num_workers=workers, collate_fn=collate,
                        persistent_workers=False, prefetch_factor=4 if workers else None)
    with torch.no_grad():
        for x, metas in loader:
            emb = net.encode_image(x.to(device))
            yield metas, torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()


def backfill_taken(db, paths, workers):
    """Fill in capture times for rows embedded before the taken column existed."""
    if not paths:
        return
    with Pool(workers) as pool:
        times = pool.map(capture_time_of, paths, chunksize=64)
    db.executemany("UPDATE images SET taken=? WHERE path=?", list(zip(times, paths)))
    db.commit()
    print(f"read capture times for {len(paths)} previously embedded images", flush=True)


def embed(root, db_path, batch=128, workers=12, limit=None, model=MODEL, pretrained=PRETRAINED):
    """Embed every new or changed still image under root into the database at db_path."""
    db = open_db(db_path)
    try:
        _embed(os.path.expanduser(root), db, batch, workers, limit, model, pretrained)
    finally:
        db.close()


def _embed(root, db, batch, workers, limit, model, pretrained):
    mid = model_id(model, pretrained)
    have = {p: (s, m, md, t) for p, s, m, md, t in
            db.execute("SELECT path, size, mtime_ns, model, taken FROM images WHERE error IS NULL")}

    t0 = time.time()
    todo, untimed = [], []
    for path in walk(root):
        st = os.stat(path)
        row = have.get(path)
        if row is None or row[:3] != (st.st_size, st.st_mtime_ns, mid):
            todo.append((path, st.st_size, st.st_mtime_ns))
        elif row[3] is None:
            untimed.append(path)
    if limit:
        todo = todo[: limit]
    print(f"walk: {len(have)} already embedded, {len(todo)} to do with {mid} ({time.time() - t0:.1f}s)", flush=True)
    backfill_taken(db, untimed, workers or 1)
    if not todo:
        return

    batches = (batches_tiny(todo, batch, workers or 1) if model == TINY
               else batches_clip(todo, batch, workers, model, pretrained))
    t1, done, errors = time.time(), 0, 0
    for metas, emb in batches:
        rows = []
        for m, e in zip(metas, emb):
            blob = None if m["error"] or e is None else np.asarray(e, np.float32).tobytes()
            errors += bool(m["error"])
            rows.append((m["path"], m["size"], m["mtime_ns"], m["sha256"] or None, m["width"], m["height"],
                         mid, blob, m["error"] or None, m["taken"]))
        db.executemany(f"INSERT OR REPLACE INTO images ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))})",
                       rows)
        db.commit()
        done += len(metas)
        if done % (batch * 20) < batch:
            rate = done / (time.time() - t1)
            eta = (len(todo) - done) / rate
            print(f"{done}/{len(todo)}  {rate:.1f} img/s  errors={errors}  eta {eta / 60:.1f} min", flush=True)
    dt = max(time.time() - t1, 1e-9)
    print(f"embedded {done} images in {dt:.0f}s ({done / dt:.1f} img/s), errors={errors}", flush=True)


def add_embed_args(ap):
    """Embedding options shared by embed_photos.py and photo_clusters.py."""
    g = ap.add_argument_group("embedding")
    g.add_argument("--model", default=MODEL, help=f"open_clip model name, or {TINY} for tests (default {MODEL})")
    g.add_argument("--pretrained", default=PRETRAINED, help=f"open_clip weights (default {PRETRAINED})")
    g.add_argument("--batch", type=int, default=128, help="images per model batch (default 128)")
    g.add_argument("--workers", type=int, default=12, help="processes reading and decoding files (default 12)")
    g.add_argument("--limit", type=int, help="embed at most this many new files (for trying things out)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--db", default="out/embeddings.sqlite3")
    add_embed_args(ap)
    args = ap.parse_args()
    embed(args.root, args.db, args.batch, args.workers, args.limit, args.model, args.pretrained)


if __name__ == "__main__":
    sys.exit(main())
