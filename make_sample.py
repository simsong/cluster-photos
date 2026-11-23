"""Generate a small synthetic photo directory for trying cluster-photos without using your own pictures.

Theory of purpose: a new user should be able to run the whole pipeline, and see clusters on the review page,
before pointing it at a real archive. No photos ship with the repository, so this script draws some.
Theory of operation: it draws a handful of distinct scenes (sky and sun, checkerboard, rings, stripes, random
blobs) and writes each several times into nested folders: the original, a half-size copy, a re-encoded JPEG, a
slightly shifted version and a PNG. Those copies should cluster together, and the half-size copies and re-encoded
JPEGs should be offered as copies of a better file. JPEGs carry an EXIF capture time. Two byte-identical backups
are added (the shorter path should be kept), and the checkerboard is "photographed again" a year later, which
the capture-time split should put in a cluster of its own. A few unrelated noise images should not cluster,
except a noise image and its byte-identical copy.
"""

import argparse
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def scene(kind, rng, size=(1200, 900)):
    w, h = size
    img = Image.new("RGB", size)
    d = ImageDraw.Draw(img)
    if kind == "sunset":
        for y in range(h):
            t = y / h
            d.line([(0, y), (w, y)], fill=(int(250 - 120 * t), int(140 - 100 * t), int(60 + 120 * t)))
        d.ellipse([w * 0.4, h * 0.45, w * 0.6, h * 0.72], fill=(255, 230, 120))
        d.rectangle([0, h * 0.7, w, h], fill=(20, 40, 70))
    elif kind == "checker":
        s = 100
        for y in range(0, h, s):
            for x in range(0, w, s):
                d.rectangle([x, y, x + s, y + s], fill=(230, 230, 230) if (x // s + y // s) % 2 else (30, 30, 30))
    elif kind == "rings":
        img.paste((10, 60, 20), [0, 0, w, h])
        for r in range(500, 0, -50):
            c = (40 + r // 3, 200 - r // 4, 80) if (r // 50) % 2 else (240, 240, 200)
            d.ellipse([w / 2 - r, h / 2 - r, w / 2 + r, h / 2 + r], fill=c)
    elif kind == "stripes":
        for x in range(0, w, 60):
            d.rectangle([x, 0, x + 30, h], fill=(200, 30, 40))
            d.rectangle([x + 30, 0, x + 60, h], fill=(250, 250, 250))
    elif kind == "blobs":
        img.paste((90, 150, 220), [0, 0, w, h])
        for _ in range(25):
            x, y, r = rng.integers(0, w), rng.integers(0, h), rng.integers(30, 160)
            d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    for _ in range(120):  # small details give the resolution check (SIFT) features to match
        x, y, r = rng.integers(0, w), rng.integers(0, h), rng.integers(6, 22)
        d.rectangle([x, y, x + r, y + r // 2], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return img.filter(ImageFilter.GaussianBlur(1))


def exif(when):
    e = Image.Exif()
    e.get_ifd(0x8769)[0x9003] = when  # DateTimeOriginal
    return e


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", nargs="?", default="sample", help="directory to create (default: sample)")
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    n = 0
    for i, kind in enumerate(("sunset", "checker", "rings", "stripes", "blobs")):
        img = scene(kind, rng)
        a, b = os.path.join(args.out, "trip", kind), os.path.join(args.out, "backup", "copies")
        os.makedirs(a, exist_ok=True)
        os.makedirs(b, exist_ok=True)
        when = exif(f"2024:06:01 10:{i:02d}:00")
        img.save(f"{a}/{kind}.jpg", quality=95, exif=when)
        img.resize((600, 450)).save(f"{a}/{kind}_small.jpg", quality=90, exif=when)
        img.save(f"{b}/{kind}_resaved.jpg", quality=60, exif=when)
        img.crop((20, 15, 1200, 900)).save(f"{b}/{kind}_shifted.jpg", quality=90, exif=when)
        img.save(f"{b}/{kind}.png")  # no capture time: it joins the piece of its most similar photo
        n += 5
        if kind == "checker":  # the same view a year later, twice
            later = os.path.join(args.out, "later")
            os.makedirs(later, exist_ok=True)
            for j, box in enumerate([(10, 10, 1190, 890), (30, 20, 1180, 880)]):
                img.crop(box).resize((1200, 900)).save(f"{later}/checker_again{j}.jpg", quality=90,
                                                       exif=exif(f"2025:06:01 09:0{j}:00"))
                n += 1
    old = os.path.join(args.out, "backup", "old laptop", "trip", "sunset")
    os.makedirs(old, exist_ok=True)
    shutil.copyfile(f"{args.out}/trip/sunset/sunset.jpg", f"{old}/sunset.jpg")
    os.makedirs(os.path.join(args.out, "misc", "dup"), exist_ok=True)
    for i in range(4):
        Image.fromarray(rng.integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(f"{args.out}/misc/noise{i}.png")
        n += 1
    shutil.copyfile(f"{args.out}/misc/noise0.png", f"{args.out}/misc/dup/noise0.png")
    n += 2
    print(f"wrote {n} images under {args.out}/")


if __name__ == "__main__":
    main()
