"""Test Target methods."""

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
from unittest.mock import MagicMock
from unittest.mock import patch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mht.target import Target


class TestTarget(unittest.TestCase):
    """Target method tests."""

    def setUp(self):
        """Set up."""
        self.filter = MagicMock()

    @patch('mht.target.Track')
    def test_initial(self, trackmock):
        """Test creation of new target."""
        t = Target.initial(self.filter, 0)

        self.assertEqual(len(t.tracks), 1)
