"""Test motion model."""

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
