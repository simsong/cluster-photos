"""Tests for the graph clustering in cluster_review.py, on synthetic unit vectors and neighbour lists.

Purpose: check that thresholded components, the recursive split, the capture-time split and the grouping of
byte-identical files behave as the README describes.
Operation: small graphs are built by hand. A chain of moderately similar nodes must split into tight groups,
while a cohesive component must stay whole, and edges below a threshold must not connect nodes. The neighbour
search runs with numpy, and also with PyTorch when it is installed.
No model, database or photos are involved, so the tests are fast and need no fixtures.
"""

import datetime
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cluster_review as cr  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


def unit(*angles):
    return np.array([[np.cos(a), np.sin(a)] for a in angles], np.float32)


def day(d, h=12):
    return datetime.datetime(2024, 6, d, h)


class Clustering(unittest.TestCase):
    def check_knn(self, M):
        nbr = cr.knn_graph(M, k=20, start=0.9)
        self.assertEqual({b for b, _ in nbr[0]}, {1})
        self.assertEqual({b for b, _ in nbr[1]}, {0})
        self.assertEqual(nbr[2], [])

    def test_knn_graph_handles_fewer_images_than_k(self):
        self.check_knn(unit(0.0, 0.02, 1.5))

    def test_knn_graph_with_numpy_only(self):
        saved = cr._torch
        cr._torch = lambda: None
        try:
            self.check_knn(unit(0.0, 0.02, 1.5))
        finally:
            cr._torch = saved

    @unittest.skipUnless(torch, "PyTorch is not installed")
    def test_knn_graph_accepts_a_torch_tensor(self):
        self.check_knn(torch.from_numpy(unit(0.0, 0.02, 1.5)))

    def test_components_respect_threshold(self):
        nbr = [[(1, 0.95)], [(0, 0.95), (2, 0.91)], [(1, 0.91)]]
        self.assertEqual(sorted(map(sorted, cr.components(range(3), nbr, 0.90))), [[0, 1, 2]])
        self.assertEqual(sorted(map(sorted, cr.components(range(3), nbr, 0.92))), [[0, 1], [2]])

    def test_components_stay_inside_node_set(self):
        nbr = [[(1, 0.99)], [(0, 0.99), (2, 0.99)], [(1, 0.99)]]
        self.assertEqual(sorted(map(sorted, cr.components([0, 2], nbr, 0.9))), [[0], [2]])

    def test_cohesive_component_is_not_split(self):
        E = unit(0.0, 0.02, 0.04)
        nbr = [[(1, 0.999)], [(0, 0.999), (2, 0.999)], [(1, 0.999)]]
        self.assertEqual(cr.split([0, 1, 2], nbr, E, 0), [[0, 1, 2]])

    def test_loose_chain_is_split_at_weak_link(self):
        E = unit(0.0, 0.05, 1.5, 1.55)  # two tight pairs far apart (min cosine to centroid 0.71), joined by one weak edge
        nbr = [[(1, 0.999)], [(0, 0.999), (2, 0.91)], [(1, 0.91), (3, 0.999)], [(2, 0.999)]]
        self.assertEqual(sorted(map(sorted, cr.split([0, 1, 2, 3], nbr, E, 0))), [[0, 1], [2, 3]])

    def test_settings_change_the_split(self):
        E = unit(0.0, 0.05, 1.5, 1.55)
        nbr = [[(1, 0.999)], [(0, 0.999), (2, 0.91)], [(1, 0.91), (3, 0.999)], [(2, 0.999)]]
        lax = cr.Settings(cohesion=0.5)  # nothing counts as loose, so the chain stays whole
        self.assertEqual(sorted(map(sorted, cr.split([0, 1, 2, 3], nbr, E, 0, lax))), [[0, 1, 2, 3]])

    def test_time_gap_splits_and_untimed_photo_follows_its_nearest(self):
        E = unit(0.0, 0.01, 0.02, 0.03, 0.025)
        taken = [day(1), day(1, 13), day(20), day(20, 9), None]  # 4 is closest to 3 (and 2)
        pieces = cr.split_by_time([0, 1, 2, 3, 4], taken, E, 24)
        self.assertEqual(sorted(map(sorted, pieces)), [[0, 1], [2, 3, 4]])

    def test_time_gap_zero_or_short_span_keeps_cluster(self):
        E = unit(0.0, 0.01, 0.02)
        self.assertEqual(cr.split_by_time([0, 1, 2], [day(1), day(20), None], E, 0), [[0, 1, 2]])
        self.assertEqual(cr.split_by_time([0, 1, 2], [day(1), day(1, 20), None], E, 24), [[0, 1, 2]])
        self.assertEqual(cr.split_by_time([0, 1], [None, None], E, 24), [[0, 1]])

    def test_identical_keeper_is_shortest_path(self):
        paths = ["/p/backup/old/a.jpg", "/p/a.jpg", "/p/b/a.jpg", "/p/x.jpg", "/p/y.jpg"]
        dup = cr.identical_groups(paths, ["h1", "h1", "h1", "h2", None])
        self.assertEqual(dup, {0: 1, 2: 1})

    def test_identical_copies_join_their_keepers_cluster(self):
        # 0 and 1 look alike; 2 is a byte copy of 0; 3 and 4 are byte copies of a photo like nothing else
        E = unit(0.0, 0.01, 0.0, 2.0, 2.0)
        paths = ["/p/a.jpg", "/p/b.jpg", "/p/backup/a.jpg", "/p/n.png", "/p/dup/n.png"]
        clusters, identical = cr.cluster(E, paths, ["h0", "h1", "h0", "h3", "h3"], [None] * 5)
        self.assertEqual(identical, {2: 0, 4: 3})
        self.assertEqual(sorted(map(sorted, clusters)), [[0, 1, 2], [3, 4]])


if __name__ == "__main__":
    unittest.main()
