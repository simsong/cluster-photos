"""Compute CLIP image embeddings for every still image under a root and store them in sqlite.

Theory of purpose: give the clustering step a perceptual signal ("these two photos look
the same") that survives re-encoding, resizing and metadata changes.
Theory of operation: walk the root (read-only), skip files whose (path, size, mtime_ns)
row is already present, and feed the rest through a DataLoader whose workers read each
file once to compute sha256 and decode it (EXIF-transposed, HEIC via pillow-heif).
The main process runs open_clip on CUDA, MPS or the CPU and stores L2-normalized float32 vectors as raw
blobs (the sqlite-vec vector format, so vec_distance_cosine() works on them directly).
"""

import argparse
import hashlib
import io
import os
import sqlite3
import sys
import time

import open_clip
import torch
from PIL import Image, ImageFile, ImageOps
from pillow_heif import register_heif_opener
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTS = {".jpg", ".jpeg", ".heic", ".png", ".gif", ".tif", ".tiff", ".webp", ".bmp"}
Image.MAX_IMAGE_PIXELS = None  # local archive: large scans/panoramas are legitimate
ImageFile.LOAD_TRUNCATED_IMAGES = True  # embed what decodes from partially written files
MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"

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
    error TEXT
);
CREATE INDEX IF NOT EXISTS images_sha ON images(sha256);
"""


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if not name.startswith(".") and os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                yield os.path.join(dirpath, name)


class Photos(Dataset):
    def __init__(self, items, preprocess):
        self.items, self.preprocess = items, preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        register_heif_opener()
        path, size, mtime_ns = self.items[i]
        meta = {"path": path, "size": size, "mtime_ns": mtime_ns, "sha256": "", "width": 0, "height": 0, "error": ""}
        try:
            with open(path, "rb") as f:
                data = f.read()
            meta["sha256"] = hashlib.sha256(data).hexdigest()
            img = Image.open(io.BytesIO(data))
            w, h = img.size  # true pixel size, before draft() downscales the decode
            meta["width"], meta["height"] = (h, w) if img.getexif().get(0x0112) in (5, 6, 7, 8) else (w, h)
            img.draft("RGB", (448, 448))  # fast JPEG DCT downscale; no-op for other formats
            img = ImageOps.exif_transpose(img)
            return self.preprocess(img.convert("RGB")), meta
        except Exception as e:  # corrupt or unsupported file: record it and move on
            meta["error"] = f"{type(e).__name__}: {e}"[:500]
            return torch.zeros(3, 224, 224), meta


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def embed(root, db_path, batch=128, workers=12, limit=None):
    """Embed every new or changed still image under root into the database at db_path."""
    root = os.path.expanduser(root)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    db = sqlite3.connect(db_path)
    db.executescript(SCHEMA)
    have = {p: (s, m) for p, s, m in db.execute("SELECT path, size, mtime_ns FROM images WHERE error IS NULL")}

    t0 = time.time()
    todo = []
    for path in walk(root):
        st = os.stat(path)
        if have.get(path) != (st.st_size, st.st_mtime_ns):
            todo.append((path, st.st_size, st.st_mtime_ns))
    if limit:
        todo = todo[: limit]
    print(f"walk: {len(have)} already embedded, {len(todo)} to do ({time.time() - t0:.1f}s)", flush=True)
    if not todo:
        return

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=device)
    model.eval()
    loader = DataLoader(Photos(todo, preprocess), batch_size=batch, num_workers=workers,
                        collate_fn=collate, persistent_workers=False, prefetch_factor=4)

    t1, done, errors = time.time(), 0, 0
    with torch.no_grad():
        for x, metas in loader:
            emb = model.encode_image(x.to(device))
            emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
            rows = []
            for m, e in zip(metas, emb):
                blob = None if m["error"] else e.tobytes()
                errors += bool(m["error"])
                rows.append((m["path"], m["size"], m["mtime_ns"], m["sha256"] or None, m["width"], m["height"],
                             f"{MODEL}/{PRETRAINED}", blob, m["error"] or None))
            db.executemany("INSERT OR REPLACE INTO images VALUES (?,?,?,?,?,?,?,?,?)", rows)
            db.commit()
            done += len(metas)
            if done % (batch * 20) < batch:
                rate = done / (time.time() - t1)
                eta = (len(todo) - done) / rate
                print(f"{done}/{len(todo)}  {rate:.1f} img/s  errors={errors}  eta {eta / 60:.1f} min", flush=True)
    dt = time.time() - t1
    print(f"embedded {done} images in {dt:.0f}s ({done / dt:.1f} img/s), errors={errors}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("root")
    ap.add_argument("--db", default="out/embeddings.sqlite3")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    embed(args.root, args.db, args.batch, args.workers, args.limit)


if __name__ == "__main__":
    sys.exit(main())
