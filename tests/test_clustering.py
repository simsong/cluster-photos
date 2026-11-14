"""Tests for the graph clustering in cluster_review.py, on synthetic unit vectors and neighbour lists.

Purpose: check that thresholded components and the recursive split behave as the README describes.
Operation: small graphs are built by hand. A chain of moderately similar nodes must split into tight groups,
while a cohesive component must stay whole, and edges below a threshold must not connect nodes.
No model, database or photos are involved, so the tests are fast and need no fixtures.
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cluster_review as cr  # noqa: E402


def unit(*angles):
    return np.array([[np.cos(a), np.sin(a)] for a in angles], np.float32)


class Clustering(unittest.TestCase):
    def test_knn_graph_handles_fewer_images_than_k(self):
        M = torch.from_numpy(unit(0.0, 0.02, 1.5))
        nbr = cr.knn_graph(M, k=20, start=0.9)
        self.assertEqual({b for b, _ in nbr[0]}, {1})
        self.assertEqual({b for b, _ in nbr[1]}, {0})
        self.assertEqual(nbr[2], [])

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


if __name__ == "__main__":
    unittest.main()
