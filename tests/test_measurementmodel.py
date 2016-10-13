"""Test measurement model."""

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
