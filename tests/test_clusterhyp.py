"""Test cluster hypothesis methods."""

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
from mht.clusterhyp import ClusterHypothesis


class TestClusterHypothesis(unittest.TestCase):
    """Test cluster hypothesis methods."""

    def setUp(self):
        """Set up."""
        self.tracker = MagicMock()
        self.tracker.k_max = 10
        self.reports = [MagicMock(), MagicMock(), MagicMock()]
        self.tracks = [MagicMock(), MagicMock(), MagicMock()]
        self.new_tracks = [MagicMock(), MagicMock(), MagicMock()]
        self.filters = [MagicMock(), MagicMock(), MagicMock()]
        self.targets = [MagicMock(), MagicMock(), MagicMock()]
        self.clusters = [MagicMock(), MagicMock(), MagicMock()]
        self.hyps = [MagicMock(), MagicMock(), MagicMock()]
        for i in range(len(self.tracks)):
            self.tracks[i].children = {self.reports[i]: self.new_tracks[i]}
            self.tracks[i].filter = [self.filters[i]]
            self.tracks[i].score.return_value = i + 2
            self.tracks[i].assign.return_value = self.new_tracks[i]
            self.new_tracks[i].parent = self.tracks[i]
            self.new_tracks[i].score.return_value = 2 * i + 2
            self.targets[i].tracks = {None: self.tracks[i]}
            self.tracks[i].target = self.targets[i]
            self.clusters[i].targets = [self.targets[i]]
            self.hyps[i].tracks = [self.tracks[i]]
            self.hyps[i].targets = [self.targets[i]]
        self.sensor = MagicMock()
        self.sensor.score_extraneous = 3
        self.sensor.score_miss = 3
        self.sensor.score_found = 0.05

    def test_initial(self):
        """Test initial."""
        chyp = ClusterHypothesis.initial(self.tracks)

        self.assertSetEqual(set(chyp.targets), set(self.targets))
        self.assertEqual(len(chyp.targets), len(self.targets))
        self.assertEqual(chyp.tracks, self.tracks)
        self.assertEqual(chyp.score(), 9)

    def test_new(self):
        """Test the creation of new hypotheses."""
        assignments = list(zip(self.reports, self.tracks))
        ph = MagicMock()
        chyp = ClusterHypothesis.new(ph, assignments, self.sensor)

        self.assertEqual(len(chyp.tracks), len(assignments))
        self.assertEqual(len(chyp.targets), len(assignments))
        self.assertEqual(chyp.score(), 12)

    def test_split(self):
        """Test hypothesis splitting."""
        chyp = ClusterHypothesis.initial(self.tracks)

        split_hyp = chyp.split({self.targets[0]})

        self.assertEqual(len(split_hyp.tracks), 1)
        self.assertEqual(split_hyp.tracks, self.tracks[0:1])
        self.assertIs(split_hyp.tracks[0].target, self.targets[0])
        self.assertEqual(split_hyp.score(), 2)

    def test_merge(self):
        """Test hypothesis merging."""
        chyps = [ClusterHypothesis.initial([tr]) for tr in self.tracks]

        merged_hyp = ClusterHypothesis.merge(chyps)

        self.assertSetEqual(set(merged_hyp.tracks), set(self.tracks))
        self.assertEqual(len(merged_hyp.tracks), len(self.tracks))
        self.assertSetEqual({tr.target
                             for tr in merged_hyp.tracks}, set(self.targets))
        self.assertSetEqual(set(merged_hyp.targets), set(self.targets))
        self.assertEqual(merged_hyp.score(), 9)
