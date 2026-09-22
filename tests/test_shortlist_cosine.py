import unittest
import numpy as np
from laya.shortlist import _cosine

class ShortlistCosineTests(unittest.TestCase):
    def test_cosine_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        docs = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]], dtype=np.float64)
        sims = _cosine(v, docs)
        self.assertAlmostEqual(sims[0], 1.0, places=5)
        self.assertAlmostEqual(sims[1], -1.0, places=5)

    def test_cosine_zero_vectors(self):
        v = np.zeros(4, dtype=np.float64)
        docs = np.ones((3, 4), dtype=np.float64)
        sims = _cosine(v, docs)
        self.assertTrue(np.all(sims == 0.0))

    def test_cosine_empty_docs(self):
        v = np.ones(4, dtype=np.float64)
        docs = np.zeros((0, 4), dtype=np.float64)
        sims = _cosine(v, docs)
        self.assertEqual(len(sims), 0)

if __name__ == "__main__":
    unittest.main()
