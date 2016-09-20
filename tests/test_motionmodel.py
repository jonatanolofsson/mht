"""Test motion model."""

import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mht


class TestCV2D(unittest.TestCase):
    """Test constant velocity update function."""

    def setUp(self):
        """Set up."""
        self.x = np.array([0] * 4)
        self.P = np.eye(4)
        self.model = mht.models.ConstantVelocityModel(0.1)

    def test_update(self):
        """Test simple update."""
        dT = 1
        x, P = self.model(self.x, self.P, dT)
        self.assertAlmostEqual(x[0], self.x[0] + self.x[2] * dT)
