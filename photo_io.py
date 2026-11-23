"""Image-reading helpers shared by the embedding, clustering, resolution and serving programs.

Theory of purpose: every program opens photos the same way (large images allowed, truncated files decoded as far
as possible, HEIC through pillow-heif) and reads the capture time the same way, so the rules live in one place.
Theory of operation: register_heif() installs the pillow-heif opener once per process, and does nothing when
pillow-heif is not installed (HEIC files then fail to open and are recorded as errors). capture_time() returns
EXIF DateTimeOriginal as "YYYY-MM-DD HH:MM:SS" local camera time, or "" when the file has none.
"""

import datetime

from PIL import ExifTags, Image, ImageFile

Image.MAX_IMAGE_PIXELS = None  # local archive: large scans/panoramas are legitimate
ImageFile.LOAD_TRUNCATED_IMAGES = True  # decode what is there from partially written files
DATETIME_ORIGINAL = {v: k for k, v in ExifTags.TAGS.items()}["DateTimeOriginal"]
_heif = None


def register_heif():
    global _heif
    if _heif is None:
        try:
            from pillow_heif import register_heif_opener
            register_heif_opener()
            _heif = True
        except ImportError:  # a declared dependency; only missing in stripped-down test environments
            _heif = False
    return _heif


def capture_time(img):
    """EXIF DateTimeOriginal of an open PIL image as 'YYYY-MM-DD HH:MM:SS', or '' if absent or unreadable."""
    try:
        v = img.getexif().get_ifd(0x8769).get(DATETIME_ORIGINAL)
        return datetime.datetime.strptime(str(v).strip()[:19], "%Y:%m:%d %H:%M:%S").isoformat(" ") if v else ""
    except (ValueError, OSError, SyntaxError):
        return ""


def capture_time_of(path):
    """capture_time() for a file path; '' if the file cannot be opened."""
    register_heif()
    try:
        with Image.open(path) as img:
            return capture_time(img)
    except Exception:
        return ""


def parse_time(s):
    """Inverse of capture_time(): a naive datetime, or None for ''/None."""
    return datetime.datetime.fromisoformat(s) if s else None
