"""Test MHT methods."""

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


class TestMHT(unittest.TestCase):
    """Test the generation of global hypotheses."""

    def setUp(self):
        """Set up testcase."""
        self.tracker = mht.MHT(initial_targets=[
            mht.kf.KFilter(
                mht.models.ConstantVelocityModel(0.1),
                np.array([0.0, 0.0, 1.0, 1.0]),
                np.eye(4)
            ),
            mht.kf.KFilter(
                mht.models.ConstantVelocityModel(0.1),
                np.array([0.0, 10.0, 1.0, -1.0]),
                np.eye(4)
            )
        ])

    def test_register_scan(self):
        """Test the generation of global hypotheses."""
        self.tracker.register_scan(mht.Scan(
            mht.sensors.EyeOfMordor(3, 12),
            [
                mht.Report(
                    np.array([8.0, 8.0]),
                    np.eye(2),
                    mht.models.position_measurement),
                mht.Report(
                    np.array([2.0, 2.0]),
                    np.eye(2),
                    mht.models.position_measurement)
            ]))

    def test_predict(self):
        """Test prediction."""
        # self.assertEqual(len(list(self.tracker.targets())), 2)
        # self.assertAlmostEqual(list(self.tracker.global_hypotheses())[0].tracks[0].filter.x[0], 0)  # noqa
        # self.assertAlmostEqual(list(self.tracker.global_hypotheses())[0].tracks[0].filter.x[1], 0)  # noqa
        # self.assertAlmostEqual(list(self.tracker.global_hypotheses())[0].tracks[1].filter.x[0], 0)  # noqa
        # self.assertAlmostEqual(list(self.tracker.global_hypotheses())[0].tracks[1].filter.x[1], 10)  # noqa

        self.tracker.predict(1)

        # self.assertEqual(len(list(self.tracker.targets())), 2)
        # self.assertAlmostEqual(list(self.tracker.global_hypotheses())[0].tracks[0].filter.x[0], 1)  # noqa
        # self.assertAlmostEqual(list(self.tracker.global_hypotheses())[0].tracks[0].filter.x[1], 1)  # noqa
        # self.assertAlmostEqual(list(self.tracker.global_hypotheses())[0].tracks[1].filter.x[0], 1)  # noqa
        # self.assertAlmostEqual(list(self.tracker.global_hypotheses())[0].tracks[1].filter.x[1], 9)  # noqa

    def test_track(self):
        """Test repeated updates from moving targets."""
        targets = [
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([0.0, 10.0, 1.0, -1.0])
        ]
        for t in range(3):
            if t > 0:
                self.tracker.predict(1)
            for t in targets:
                t[0:2] += t[2:]

            reports = {mht.Report(
                np.random.multivariate_normal(t[0:2], np.diag([0.1, 0.1])),  # noqa
                # t[0:2],
                np.eye(2) * 0.001,
                mht.models.position_measurement,
                i)
                for i, t in enumerate(targets)}
            self.tracker.register_scan(mht.Scan(
                mht.sensors.EyeOfMordor(3, 12), reports))


if __name__ == '__main__':
    unittest.main()
