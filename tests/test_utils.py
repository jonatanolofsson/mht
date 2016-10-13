"""Test utility functions."""

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
