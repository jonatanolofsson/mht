"""Test Track methods."""

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

    def test_new(self):
        """Test creation of new track."""
        tr = Track.new(self.target, self.filter, 0, None)

        self.assertEqual(tr.target, self.target)
        self.assertIs(tr.parent, None)
        self.assertEqual(tr.filter, self.filter)
        self.assertEqual(tr.score(), 0)

    def test_extend(self):
        """Test extending existing track."""
        self.filter.correct = MagicMock(return_value=1)
        root = Track.new(self.target, self.filter, 10, None)
        report = MagicMock()

        tr = root.extend(report)

        self.assertEqual(tr.target, self.target)
        self.assertEqual(tr.parent, root)
        self.assertIsNot(tr.filter, self.filter)
        self.assertEqual(tr.score(), 11)
