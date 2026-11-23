"""Tests for resolution_dupes.py on real generated images: same picture resized versus crops and other pictures.

Purpose: "keep highest resolution" may delete files, so false matches matter more than missed ones.
Operation: a textured random image is saved at two sizes, and also as a crop, a different picture and a lightly
shifted frame. same_image must accept only the true resize. Name rules and keeper selection are checked directly,
including that a chain of matches cannot mark the keeper of another file. Files live in a temp directory.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import resolution_dupes as rd  # noqa: E402


def picture(seed, size=(1200, 800)):
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (40, 60, 3), dtype=np.uint8), "RGB").resize(size, Image.Resampling.BICUBIC)
    a = np.asarray(img).copy()
    for _ in range(150):  # hard-edged shapes give SIFT stable features
        x, y = rng.integers(0, size[0] - 80), rng.integers(0, size[1] - 80)
        a[y:y + rng.integers(15, 70), x:x + rng.integers(15, 70)] = rng.integers(0, 255, 3)
    return Image.fromarray(a)


class SameImage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = cls.tmp.name
        big = picture(1)
        big.save(f"{cls.d}/big.png")
        big.resize((600, 400)).save(f"{cls.d}/small.jpg", quality=92)
        big.crop((0, 0, 700, 500)).save(f"{cls.d}/crop.png")
        picture(2).save(f"{cls.d}/other.png")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def same(self, a, b):
        return rd.same_image((f"{self.d}/{a}", f"{self.d}/{b}"))[1]

    def test_resize_is_same_image(self):
        self.assertTrue(self.same("big.png", "small.jpg"))

    def test_crop_is_not_same_image(self):
        self.assertFalse(self.same("big.png", "crop.png"))

    def test_other_picture_is_not_same_image(self):
        self.assertFalse(self.same("big.png", "other.png"))


class Rules(unittest.TestCase):
    def test_frame_numbers(self):
        for a, b, want in [("DSCF0293.JPG", "DSCF0294.JPG", True), ("CCI_2.jpeg", "CCI_5.jpeg", True), ("IMG_0055.jpg", "IMG_0055 (2).jpg", False),
                             ("DSC_0145-SMILE.jpg", "DSC_0152.NEF.jpg", True), ("DSC_0145-SMILE.jpg", "DSC_0145.NEF.jpg", False)]:
            self.assertEqual(rd.different_frames(a, b), want, (a, b))

    def test_largest_is_keeper(self):
        self.assertEqual(rd.choose_copies({0: 400, 1: 100, 2: 25}, {frozenset((0, 1)), frozenset((0, 2))}), {1: 0, 2: 0})

    def test_chain_needs_a_direct_match_to_a_keeper(self):
        # 2 matches only 1, and 1 is itself a lower-res copy of 0, so 2 must stay
        self.assertEqual(rd.choose_copies({0: 400, 1: 100, 2: 25}, {frozenset((0, 1)), frozenset((1, 2))}), {1: 0})

    def test_equal_sizes_are_left_alone_without_file_sizes(self):
        self.assertEqual(rd.choose_copies({0: 100, 1: 100}, {frozenset((0, 1))}), {})

    def test_same_size_resave_keeps_larger_file(self):
        self.assertEqual(rd.choose_copies({0: 100, 1: 100}, {frozenset((0, 1))}, {0: 500, 1: 900}), {0: 1})

    def test_identical_size_and_bytes_are_left_alone(self):
        self.assertEqual(rd.choose_copies({0: 100, 1: 100}, {frozenset((0, 1))}, {0: 500, 1: 500}), {})

    def test_resave_and_lower_res_share_one_keeper(self):
        # 0 is the best file; 1 is a smaller re-save of it; 2 is half size and matches both
        pairs = {frozenset((0, 1)), frozenset((0, 2)), frozenset((1, 2))}
        self.assertEqual(rd.choose_copies({0: 400, 1: 400, 2: 100}, pairs, {0: 900, 1: 300, 2: 200}), {1: 0, 2: 0})

    def test_candidate_pairs(self):
        px, nb = {0: 400, 1: 400, 2: 100, 3: 400}, {0: 900, 1: 300, 2: 200, 3: 900}
        self.assertTrue(rd.candidate(0, 1, px, nb))   # re-save
        self.assertTrue(rd.candidate(0, 2, px, nb))   # lower resolution
        self.assertFalse(rd.candidate(0, 3, px, nb))  # same size, same bytes: identical files are handled by sha256


if __name__ == "__main__":
    unittest.main()
