"""Test Kalman Filter Target class."""

import unittest
import numpy as np

import mht


class TestKFTarget(unittest.TestCase):
    """Testcases for KF Target."""

    def setUp(self):
        """Set up."""
        model = mht.models.constant_velocity_2d(0.1)
        self.x0 = np.matrix([[0.0]] * 4)
        self.P0 = np.eye(4)
        self.target = mht.kf.Target(model, self.x0, self.P0)

    def test_predict(self):
        """Predict step."""
        dT = 1
        self.target.predict(dT)
        self.assertAlmostEqual(self.target.x[0], self.x0[0] + self.x0[2] * dT)

    def test_correct(self):
        """Correction step."""
        z = np.matrix([[2.0]] * 2)
        R = np.eye(2)
        m = mht.Measurement(z, R, mht.models.velocity_measurement)
        self.target.correct(m)
        self.assertAlmostEqual(self.target.x[1], 0.0)
        self.assertAlmostEqual(self.target.x[2], 1.0)
        self.assertAlmostEqual(self.target.x[3], 1.0)
