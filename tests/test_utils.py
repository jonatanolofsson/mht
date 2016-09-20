"""Test utility functions."""

import unittest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mht.utils


class TestUtils(unittest.TestCase):
    """Testshell for utilities."""

    def test_overlap_p_1(self):
        """Test the overlap_p function."""
        a = (0, 1, 0, 1)
        b = (0.2, 2, 0.2, 2)
        res = mht.utils.overlap_pa(a, b)

        self.assertAlmostEqual(res, 0.64)

    def test_overlap_p_2(self):
        """Test the overlap_p function."""
        a = (0, 1, 0, 1)
        b = (-0.2, 2, -0.2, 2)
        res = mht.utils.overlap_pa(a, b)

        self.assertAlmostEqual(res, 1)

    def test_overlap_p_3(self):
        """Test the overlap_p function."""
        a = (0, 1, 0, 1)
        b = (0.2, 0.8, 0.2, 0.8)
        res = mht.utils.overlap_pa(a, b)

        self.assertAlmostEqual(res, 0.36)
