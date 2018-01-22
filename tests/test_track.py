"""Test Track methods."""

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
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mht.track import Track


class TestTrack(unittest.TestCase):
    """Track method tests."""

    def setUp(self):
        """Set up."""
        self.target = MagicMock()
        self.parent = MagicMock()
        self.filter = MagicMock()
        self.sensor = MagicMock()

    def test_new(self):
        """Test creation of new track."""
        report = MagicMock()
        self.sensor.score_extraneous = 5
        tr = Track.new(self.target, self.filter, self.sensor, report)

        self.assertEqual(tr.target, self.target)
        self.assertIs(tr.parent_id, None)
        self.assertEqual(tr.filter, self.filter)
        self.assertEqual(tr.score(), 5)

    def test_extend(self):
        """Test extending existing track."""
        self.filter.correct = MagicMock(return_value=1)
        root_report = MagicMock()
        self.sensor.score_extraneous = 10
        self.sensor.score_miss = 0
        self.sensor.score_found = 0
        root = Track.new(self.target, self.filter, self.sensor, root_report)
        root._id = 0
        report = MagicMock()

        tr = Track.extend(root, report, self.sensor)

        self.assertEqual(tr.target, self.target)
        self.assertEqual(tr.parent_id, 0)
        self.assertIsNot(tr.filter, self.filter)
        # self.assertEqual(tr.score(), 11)
