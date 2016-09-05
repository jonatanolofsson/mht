"""Test Target methods."""

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

        trackmock.new.assert_called_once_with(t, self.filter, 0, None)
        self.assertEqual(len(t.tracks), 1)
