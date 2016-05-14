"""Test measurement model."""

import unittest
import numpy as np

import mht


class TestMV2D(unittest.TestCase):
    """Test constant velocity update function."""

    def setUp(self):
        """Set up."""
        self.x = np.matrix([[0]] * 4)
        self.mfn = mht.models.velocity_measurement

    def test_measure(self):
        """Test simple update."""
        z, H = self.mfn(self.x)
        self.assertAlmostEqual(z[0], 0)
