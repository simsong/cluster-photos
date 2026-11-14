"""Find the same image stored at different resolutions inside a cluster, and pick the highest-resolution keeper.

Theory of purpose: clusters often hold one photo at several sizes. A person should be able to keep the largest
and discard the rest in one action, without risking crops, bursts or merely similar photos.
Theory of operation: pairs inside a cluster with CLIP cosine >= MIN_COS and pixel counts differing by at least
MIN_RATIO are checked. Different frame numbers in one naming scheme, and capture times more than 2 s apart,
are rejected. The rest are registered (SIFT, ratio test, RANSAC similarity) on reduced images, and must have at
least MIN_INLIERS matches, mutual overlap >= MIN_OVERLAP (so a crop never counts) and aligned correlation >=
MIN_ALIGNED. Files are then visited from largest to smallest: one is lower-res only if it directly matches a
larger file that is itself a keeper, so chains of matches cannot remove the last copy. Results are cached by path and size.
"""

import datetime
import json
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import ExifTags, Image, ImageFile, ImageOps
from pillow_heif import register_heif_opener

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MIN_COS, MIN_RATIO, MIN_INLIERS, MIN_ALIGNED, MIN_OVERLAP, WORK = 0.97, 1.05, 30, 0.95, 0.85, 1000
SEQ = re.compile(r"^(.*?[A-Za-z_ -])(\d{3,})\.(jpe?g|png|tiff?|heic)$", re.I)
FRAME = re.compile(r"^([A-Za-z_]+)(\d{3,})(?!\d)")  # the camera's frame number, whatever follows it
TAG = {v: k for k, v in ExifTags.TAGS.items()}


def different_frames(a, b):
    """Numbered frames in one naming scheme (DSCF0293/DSCF0294, CCI_2/CCI_5, 00.00.00-1/-2) are different photos."""
    na, nb = os.path.basename(a), os.path.basename(b)
    fa, fb = FRAME.match(na), FRAME.match(nb)  # DSC_0145-SMILE.jpg vs DSC_0152.NEF.jpg
    if fa and fb and fa.group(1).lower() == fb.group(1).lower() and int(fa.group(2)) != int(fb.group(2)):
        return True
    suf = lambda n: re.search(r"(?:[_-]| \()(\d+)\)?(\.[^.]+)$", n)
    sa, sb = suf(na), suf(nb)
    if sa and sb and sa.group(1) != sb.group(1) and na[:sa.start()].lower() == nb[:sb.start()].lower():
        return True
    strip = lambda n: re.sub(r"( \(\d+\)|_\d|-\d| copy)+(\.[^.]+)$", r"\2", n, flags=re.I)
    x, y = SEQ.match(strip(na)), SEQ.match(strip(nb))
    return bool(x and y and x.group(1).lower() == y.group(1).lower() and int(x.group(2)) != int(y.group(2)))


def is_burst(a, b):
    """True when both files record capture times that differ by more than 2 s, apart from whole-hour time zone shifts."""
    def stamp(p):
        v = Image.open(p).getexif().get_ifd(0x8769).get(TAG["DateTimeOriginal"])
        return datetime.datetime.strptime(str(v).strip()[:19], "%Y:%m:%d %H:%M:%S") if v else None
    try:
        register_heif_opener()
        x, y = stamp(a), stamp(b)
    except (OSError, ValueError):
        return False
    if not (x and y):
        return False
    d = abs((x - y).total_seconds())
    return d > 2 and not (d <= 14 * 3600 and min(d % 3600, 3600 - d % 3600) <= 2)


def _gray(path):
    register_heif_opener()
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    img.thumbnail((WORK, WORK))
    return np.asarray(img)


def same_image(pair):
    """Return ((a, b), True) when a and b are the same picture at any scale, uncropped."""
    a, b = pair
    cv2.setNumThreads(1)
    try:
        if different_frames(a, b) or is_burst(a, b):
            return pair, False
        ga, gb = _gray(a), _gray(b)
        sift = cv2.SIFT_create(3000)
        ka, da = sift.detectAndCompute(ga, None)
        kb, db = sift.detectAndCompute(gb, None)
        if da is None or db is None or len(ka) < 8 or len(kb) < 8:
            return pair, False
        good = [m for m, n in (p for p in cv2.BFMatcher().knnMatch(da, db, k=2) if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            return pair, False
        src = np.float32([ka[m.queryIdx].pt for m in good])
        dst = np.float32([kb[m.trainIdx].pt for m in good])
        H, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if H is None or inl.sum() < MIN_INLIERS:
            return pair, False
        size = (gb.shape[1], gb.shape[0])
        warped = cv2.warpAffine(ga.astype(np.float32), H, size)
        cover = cv2.warpAffine(np.ones(ga.shape, np.uint8), H, size, flags=cv2.INTER_NEAREST).astype(bool)
        back = cv2.warpAffine(np.ones(gb.shape, np.uint8), cv2.invertAffineTransform(H), (ga.shape[1], ga.shape[0]),
                              flags=cv2.INTER_NEAREST)
        if cover.mean() < MIN_OVERLAP or back.mean() < MIN_OVERLAP:
            return pair, False
        return pair, bool(np.corrcoef(warped[cover], gb[cover].astype(np.float32))[0, 1] >= MIN_ALIGNED)
    except Exception:
        return pair, False


def choose_lower_res(n_pixels, matches):
    """Map each lower-res node to its keeper; matches is a set of frozenset({i, j}) verified pairs."""
    lower = {}
    for x in sorted(n_pixels, key=lambda i: -n_pixels[i]):
        bigger = [y for y in n_pixels if y != x and frozenset((x, y)) in matches
                  and n_pixels[y] >= MIN_RATIO * n_pixels[x] and y not in lower]
        if bigger:
            lower[x] = max(bigger, key=lambda y: n_pixels[y])
    return lower


def find_lower_res(clusters, paths, E, pixels, cache_path, workers=12):
    """Return {index: keeper index} over the given clusters of indices into paths."""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    key = lambda a, b: "\t".join(f"{paths[i]}|{os.path.getsize(paths[i])}" for i in sorted((a, b)))
    todo, pairs = [], []
    for c in clusters:
        idx = np.array(c)
        S = E[idx] @ E[idx].T
        for i, j in zip(*np.nonzero(np.triu(S >= MIN_COS, 1))):
            a, b = int(idx[i]), int(idx[j])
            hi, lo = max(pixels[a], pixels[b]), min(pixels[a], pixels[b])
            if lo and hi >= MIN_RATIO * lo:
                pairs.append((a, b))
                if key(a, b) not in cache:
                    todo.append((a, b))
    if todo:
        with Pool(workers) as pool:
            done = pool.map(same_image, [(paths[a], paths[b]) for a, b in todo], chunksize=4)
        for (a, b), (_, ok) in zip(todo, done):
            cache[key(a, b)] = ok
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        json.dump(cache, open(cache_path, "w"))
    matches = {frozenset(p) for p in pairs if cache[key(*p)]}
    lower = {}
    for c in clusters:
        lower.update(choose_lower_res({i: pixels[i] for i in c}, matches))
    return lower
