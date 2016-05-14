"""Test motion model."""

import unittest
import numpy as np

import mht


class TestCV2D(unittest.TestCase):
    """Test constant velocity update function."""

    def setUp(self):
        """Set up."""
        self.x = np.matrix([[0]] * 4)
        self.P = np.eye(4)
        self.model = mht.models.constant_velocity_2d(0.1)

    def test_update(self):
        """Test simple update."""
        dT = 1
        x, P = self.model(self.x, self.P, dT)
        self.assertAlmostEqual(x[0], self.x[0] + self.x[2] * dT)
