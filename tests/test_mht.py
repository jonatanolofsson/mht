"""Test MHT methods."""

import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mht


class TestMHT(unittest.TestCase):
    """Test the generation of global hypotheses."""

    def setUp(self):
        """Set up testcase."""
        self.tracker = mht.MHT(initial_targets=[
            mht.Target(mht.kf.KFilter(
                mht.models.constant_velocity_2d(0.1),
                np.matrix([[0.0], [0.0], [1.0], [1.0]]),
                np.eye(4)
            ), 0),
            mht.Target(mht.kf.KFilter(
                mht.models.constant_velocity_2d(0.1),
                np.matrix([[0.0], [10.0], [1.0], [-1.0]]),
                np.eye(4)
            ), 0)
        ])

    def test_register_scan(self):
        """Test the generation of global hypotheses."""
        self.tracker.register_scan(mht.Scan(
            mht.sensors.EyeOfMordor(5, 10, 12),
            [
                mht.Report(
                    np.matrix([[8.0], [8.0]]),
                    np.eye(2),
                    mht.models.position_measurement),
                mht.Report(
                    np.matrix([[2.0], [2.0]]),
                    np.eye(2),
                    mht.models.position_measurement)
            ]))
        print(self.tracker.global_hypotheses)

    def test_predict(self):
        """Test prediction."""
        self.assertEqual(len(self.tracker.targets), 2)
        print(self.tracker.global_hypotheses)
        self.assertAlmostEqual(list(self.tracker.global_hypotheses.values())[0].tracks[0].filter.x[0], 0)  # noqa
        self.assertAlmostEqual(list(self.tracker.global_hypotheses.values())[0].tracks[0].filter.x[1], 0)  # noqa
        self.assertAlmostEqual(list(self.tracker.global_hypotheses.values())[0].tracks[1].filter.x[0], 0)  # noqa
        self.assertAlmostEqual(list(self.tracker.global_hypotheses.values())[0].tracks[1].filter.x[1], 10)  # noqa

        self.tracker.predict(1)

        self.assertEqual(len(self.tracker.targets), 2)
        self.assertAlmostEqual(list(self.tracker.global_hypotheses.values())[0].tracks[0].filter.x[0], 1)  # noqa
        self.assertAlmostEqual(list(self.tracker.global_hypotheses.values())[0].tracks[0].filter.x[1], 1)  # noqa
        self.assertAlmostEqual(list(self.tracker.global_hypotheses.values())[0].tracks[1].filter.x[0], 1)  # noqa
        self.assertAlmostEqual(list(self.tracker.global_hypotheses.values())[0].tracks[1].filter.x[1], 9)  # noqa

    def test_track(self):
        """Test repeated updates from moving targets."""
        target_1 = np.matrix([[0.0], [0.0], [1.0], [1.0]])
        target_2 = np.matrix([[0.0], [10.0], [1.0], [-1.0]])
        for t in range(10):
            if t > 0:
                self.tracker.predict(1)
                target_1[0:2] += target_1[2:]
                target_2[0:2] += target_2[2:]

            self.tracker.register_scan(mht.Scan(
                mht.sensors.EyeOfMordor(5, 10, 12),
                [
                    mht.Report(
                        target_1[0:2] + np.random.normal(size=(2, 1)) * 0.3,
                        np.eye(2),
                        mht.models.position_measurement),
                    mht.Report(
                        target_2[0:2] + np.random.normal(size=(2, 1)) * 0.3,
                        np.eye(2),
                        mht.models.position_measurement)
                ]))


if __name__ == '__main__':
    unittest.main()
