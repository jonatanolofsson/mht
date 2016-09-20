"""Test measurement model."""

import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mht


class TestMV2D(unittest.TestCase):
    """Test constant velocity update function."""

    def setUp(self):
        """Set up."""
        self.x = np.array([0] * 4)
        self.mfn = mht.models.position_measurement

    def test_measure(self):
        """Test simple update."""
        z, H = self.mfn(self.x)
        self.assertAlmostEqual(z[0], 0)
