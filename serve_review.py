"""Serve a cluster review page and its photos on localhost, showing originals full size in any browser.

Theory of purpose: browsers restrict what a local file:// page may load, and Chrome cannot decode HEIC at all.
A small server avoids both limits without storing any converted copies.
Theory of operation: the output directory (page and thumbnails) is served statically. GET /photo/<id> looks the id
up in clusters.csv, so only photos on the review can be fetched and no path comes from the client. JPEG, PNG and
GIF bytes are sent unchanged; HEIC and TIFF are converted to JPEG in memory at full resolution for each request.
It binds to 127.0.0.1 and rejects other Host headers, so web pages cannot reach it through DNS rebinding.
It never writes to the photos. Marks stay in the browser, and Copy JSON still feeds apply_review.py.
"""

import argparse
import csv
import functools
import io
import mimetypes
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image, ImageFile, ImageOps

from photo_io import register_heif

Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
CONVERT = {".heic", ".heif", ".tif", ".tiff"}


def load_photos(out):
    with open(os.path.join(out, "clusters.csv"), newline="") as f:
        return {r["id"]: r["path"] for r in csv.DictReader(f)}


def photo_bytes(path):
    """Return (bytes, content type) for a photo, converting formats browsers cannot display."""
    if os.path.splitext(path)[1].lower() not in CONVERT:
        with open(path, "rb") as f:
            return f.read(), mimetypes.guess_type(path)[0] or "application/octet-stream"
    register_heif()
    buf = io.BytesIO()
    ImageOps.exif_transpose(Image.open(path)).convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue(), "image/jpeg"


class Handler(SimpleHTTPRequestHandler):
    photos, port = {}, 0

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.headers.get("Host") not in (f"127.0.0.1:{self.port}", f"localhost:{self.port}"):
            self.send_error(403)
        elif self.path == "/":
            self.send_response(302)
            self.send_header("Location", "/cluster_review.html")
            self.end_headers()
        elif self.path.startswith("/photo/"):
            self.send_photo(self.path[len("/photo/"):])
        else:
            super().do_GET()

    def send_photo(self, pid):
        path = self.photos.get(pid)
        if not path or not os.path.exists(path):
            self.send_error(404)
            return
        try:
            data, ctype = photo_bytes(path)
        except Exception:
            self.send_error(415, "could not decode this photo")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_server(out, port):
    server = ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=os.path.abspath(out)))
    Handler.photos, Handler.port = load_photos(out), server.server_address[1]
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="out/clusters", help="directory holding cluster_review.html and clusters.csv")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = make_server(args.out, args.port)
    print(f"http://127.0.0.1:{server.server_address[1]}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
