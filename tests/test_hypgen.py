"""Hypgen tests."""
import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mht.hypgen import murty, permgen, lap

MURTY_COST = np.matrix([[7, 51, 52, 87, 38, 60, 74, 66, 0, 20],
                        [50, 12, 0, 64, 8, 53, 0, 46, 76, 42],
                        [27, 77, 0, 18, 22, 48, 44, 13, 0, 57],
                        [62, 0, 3, 8, 5, 6, 14, 0, 26, 39],
                        [0, 97, 0, 5, 13, 0, 41, 31, 62, 48],
                        [79, 68, 0, 0, 15, 12, 17, 47, 35, 43],
                        [76, 99, 48, 27, 34, 0, 0, 0, 28, 0],
                        [0, 20, 9, 27, 46, 15, 84, 19, 3, 24],
                        [56, 10, 45, 39, 0, 93, 67, 79, 19, 38],
                        [27, 0, 39, 53, 46, 24, 69, 46, 23, 1]])


class TestLap(unittest.TestCase):
    """Test LAP solver."""

    def test_lap(self):
        """Test LAP solver."""
        res = lap(MURTY_COST)
        self.assertAlmostEqual(
            MURTY_COST[range(len(res[1])), res[1]].sum(),
            res[0])


class TestMurty(unittest.TestCase):
    """Test Murty algorithm."""

    def test_murty(self):
        """Test murty algo."""
        pre_res = None
        n = 0
        for res in murty(MURTY_COST):
            # print('res:', res)
            self.assertAlmostEqual(
                MURTY_COST[range(len(res[1])), res[1]].sum(),
                res[0])
            if pre_res is not None:
                self.assertGreaterEqual(res[0], pre_res[0])
            pre_res = res
            n += 1
        self.assertEqual(n, 3628800)

    def test_murty_asym(self):
        """Test asymmetric inputs for murty."""
        pre_res = None
        n = 0
        for res in murty(MURTY_COST[:5, :]):
            self.assertAlmostEqual(
                MURTY_COST[range(len(res[1])), res[1]].sum(), res[0])
            if pre_res is not None:
                self.assertGreaterEqual(res[0], pre_res[0])
            pre_res = res
            n += 1
        self.assertEqual(n, 30240)

    def test_murty_asym_small(self):
        """Test asymmetric inputs for murty."""
        pre_res = None
        n = 0
        # print(MURTY_COST[:2, :]
        for res in murty(MURTY_COST[:2, :]):
            # print(res)
            self.assertAlmostEqual(
                MURTY_COST[range(len(res[1])), res[1]].sum(), res[0])
            if pre_res is not None:
                self.assertGreaterEqual(res[0], pre_res[0])
            pre_res = res
            n += 1
        self.assertEqual(n, 90)


class TestPermgen(unittest.TestCase):
    """Test permutation generation."""

    def test_permgen(self):
        """Test permgen function."""
        D = [[(1, 'a'), (1, 'b'), (2, 'c')],
             [(1, 'd'), (2, 'e'), (3, 'f')],
             [(3, 'g')]]
        k = 0
        for res in permgen(D):
            k += 1
        self.assertEqual(k, 9)


if __name__ == '__main__':
    unittest.main()
