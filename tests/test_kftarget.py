"""Test Kalman Filter Target class."""

"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mht


class TestKFTarget(unittest.TestCase):
    """Testcases for KF Target."""

    def setUp(self):
        """Set up."""
        model = mht.models.ConstantVelocityModel(0.1)
        self.x0 = np.array([0.0] * 4)
        self.P0 = np.eye(4)
        self.target = mht.kf.KFilter(model, self.x0, self.P0)

    def test_predict(self):
        """Predict step."""
        dT = 1
        self.target.predict(dT)
        self.assertAlmostEqual(self.target.x[0], self.x0[0] + self.x0[2] * dT)

    def test_correct(self):
        """Correction step."""
        z = np.array([2.0] * 2)
        R = np.eye(2)
        m = mht.Report(z, R, mht.models.velocity_measurement)
        self.target.correct(m)
        self.assertAlmostEqual(self.target.x[1], 0.0)
        self.assertAlmostEqual(self.target.x[2], 1.0)
        self.assertAlmostEqual(self.target.x[3], 1.0)
