"""Tests for serve_review.py against a real server on an ephemeral localhost port.

Purpose: photos must be served correctly, and only photos named in clusters.csv may be reachable.
Operation: a temp directory holds generated images and a clusters.csv. Each test makes real HTTP requests with
http.client, so the Host check, the id lookup, unchanged JPEG bytes and in-memory TIFF conversion are all exercised.
No mocks are used, and the server is shut down after the class.
"""

import http.client
import io
import os
import sys
import tempfile
import threading
import unittest

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import serve_review as sr  # noqa: E402


class Serve(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.tmp.name
        Image.new("RGB", (50, 40), (200, 10, 10)).save(f"{d}/a.jpg")
        Image.new("RGB", (70, 30), (10, 200, 10)).save(f"{d}/b.tif")
        open(f"{d}/secret.txt", "w").write("x")
        os.makedirs(f"{d}/out")
        open(f"{d}/out/clusters.csv", "w").write(f"cluster,id,path\n1,aaa,{d}/a.jpg\n1,bbb,{d}/b.tif\n1,gone,{d}/nope.jpg\n")
        open(f"{d}/out/cluster_review.html", "w").write("<html>page</html>")
        cls.srv = sr.make_server(f"{d}/out", 0)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    def get(self, path, host=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port)
        c.request("GET", path, headers={"Host": host or f"127.0.0.1:{self.port}"})
        r = c.getresponse()
        body = r.read()
        c.close()
        return r.status, r.getheader("Content-Type"), body

    def test_jpeg_is_sent_unchanged(self):
        s, t, body = self.get("/photo/aaa")
        self.assertEqual((s, t), (200, "image/jpeg"))
        self.assertEqual(body, open(f"{self.tmp.name}/a.jpg", "rb").read())

    def test_tiff_is_converted_at_full_size(self):
        s, t, body = self.get("/photo/bbb")
        self.assertEqual((s, t), (200, "image/jpeg"))
        self.assertEqual(Image.open(io.BytesIO(body)).size, (70, 30))

    def test_only_listed_existing_photos(self):
        self.assertEqual(self.get("/photo/unknown")[0], 404)
        self.assertEqual(self.get("/photo/gone")[0], 404)
        self.assertEqual(self.get("/photo/../secret.txt")[0], 404)

    def test_files_outside_output_are_not_served(self):
        self.assertNotEqual(self.get("/../secret.txt")[2], b"x")

    def test_other_hosts_are_rejected(self):
        self.assertEqual(self.get("/photo/aaa", host="evil.example")[0], 403)

    def test_root_redirects_to_page(self):
        self.assertEqual(self.get("/")[0], 302)


if __name__ == "__main__":
    unittest.main()
