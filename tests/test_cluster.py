"""Test cluster methods."""

import unittest
from unittest.mock import MagicMock, call
from unittest.mock import patch
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mht
from mht.cluster import Cluster


class TestClusterInit(unittest.TestCase):
    """Test cluster class methods."""

    @patch('mht.cluster.Target')
    @patch('mht.cluster.ClusterHypothesis')
    def test_initial(self, chmock, tmock):
        """Test init."""
        tracker = MagicMock()
        targets = [MagicMock(), MagicMock()]
        tracks = [MagicMock(), MagicMock()]
        filters = [MagicMock(), MagicMock()]
        for i, t in enumerate(targets):
            t.tracks = {None: tracks[i]}
        hyp = MagicMock()
        chmock.initial = MagicMock(return_value=hyp)
        tmock.initial = MagicMock(side_effect=targets)

        cluster = Cluster.initial(tracker, filters)

        self.assertEqual(len(cluster.hypotheses), 1)
        tmock.initial.assert_has_calls([call(f) for f in filters])
        chmock.initial.assert_called_once_with(cluster, tracks)


class TestClustering(unittest.TestCase):
    """Test clustering algorithms."""

    def setUp(self):
        """Set up."""
        self.tracker = MagicMock()
        self.tracker.k_max = 10
        self.tracks = [MagicMock(), MagicMock(), MagicMock()]
        self.filters = [MagicMock(), MagicMock(), MagicMock()]
        for i, tr in enumerate(self.tracks):
            tr.filter = [self.filters[i]]
            tr.score.return_value = i + 2
        self.targets = [MagicMock(), MagicMock(), MagicMock()]
        for i, t in enumerate(self.targets):
            t.tracks = {None: self.tracks[i]}
            self.tracks[i].target = t
        self.clusters = [MagicMock(), MagicMock(), MagicMock()]
        for i, c in enumerate(self.clusters):
            c.targets = [self.targets[i]]
        self.hyps = [MagicMock(), MagicMock(), MagicMock()]
        for i, h in enumerate(self.hyps):
            h.tracks = [self.tracks[i]]
            h.targets = [self.targets[i]]

    @patch('mht.cluster.permgen')
    @patch('mht.cluster.ClusterHypothesis')
    def test_cluster_merging(self, chmock, permgen):
        """Test cluster merging."""
        merged_hyp = MagicMock()
        merged_hyp.targets = self.targets
        chmock.merge = MagicMock(return_value=merged_hyp)
        permgen.return_value = [(self.hyps, None)]

        merged_cluster = Cluster.merge(self.tracker, self.clusters)

        self.assertEqual(set(self.targets), set(merged_cluster.targets))
        chmock.merge.assert_called_once_with(merged_cluster, self.hyps)
        self.assertEqual(len(merged_cluster.hypotheses), 1)

    def test_cluster_merged_targets(self):
        """Test cluster merging."""
        tracker = mht.MHT(initial_targets=[
            mht.kf.KFilter(
                mht.models.constant_velocity_2d(0.1),
                np.matrix([[0.0], [0.0], [1.0], [1.0]]),
                np.eye(4)
            ),
            mht.kf.KFilter(
                mht.models.constant_velocity_2d(0.1),
                np.matrix([[0.0], [10.0], [1.0], [-1.0]]),
                np.eye(4)
            ),
        ])

        self.assertEqual(len(tracker.clusters), 2)

        merged_cluster = Cluster.merge(tracker, tracker.clusters)

        self.assertEqual(len(merged_cluster.targets), 2)

    @patch('mht.cluster.Target')
    @patch('mht.cluster.ClusterHypothesis')
    def test_cluster_splitting(self, chmock, tmock):
        """Test cluster splitting."""
        tmock.initial = MagicMock(side_effect=self.targets)
        chmock.initial = MagicMock(side_effect=self.hyps)
        merged_cluster = Cluster.initial(self.tracker, self.filters)
        merged_cluster.ambiguous_tracks = [set(self.tracks[0:2])]

        split_clusters = merged_cluster.split()

        self.assertEqual(len(split_clusters), 2)
        for c in split_clusters:
            self.assertEqual(len(c.hypotheses), 1)
