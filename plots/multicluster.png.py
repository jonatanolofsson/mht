"""Create crosstrack.png plot."""

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

import os
import sys
import argparse
import numpy as np
from functools import reduce
import operator
import matplotlib.pyplot as plt
# import cProfile

sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
import mht


np.random.seed(2)


def draw():
    """Create plot."""
    tracker = mht.MHT(
        cparams=mht.ClusterParameters(k_max=100, hp_limit=5),
        matching_algorithm="naive")
    target_centroids = [
        ([0.0, 0.0, 1.0, 1.0], np.diag([1, 1, 0.3, 0.3]), 15),
        ([100.0, 100.0, -1.0, -1.0], np.diag([1, 1, 0.3, 0.3]), 5),  # noqa
        ([0.0, 70.0, 1.0, 0.0], np.diag([1, 1, 0.3, 0.3]), 0),
        ([50.0, 50.0, 0.0, 0.0], np.diag([4, 4, 0.3, 0.3]), 0),  # noqa
    ]
    clutter_centroids = [
        ([0.0, 0.0, 50.0, 50.0], np.diag([70, 70, 1, 1]), 1),
    ]
    sensors = [
        mht.sensors.Satellite((-10, 40, -10, 40), 3, 3),
        mht.sensors.Satellite((45, 120, 0, 40), 3, 3),
        mht.sensors.Satellite((35, 120, 45, 120), 3, 3),
        mht.sensors.Satellite((-10, 30, 55, 120), 3, 3),
    ]

    targets = [np.random.multivariate_normal(c[0], c[1])
               for c in target_centroids for _ in range(c[2])]
    mlhyp = None
    nclusters = []
    ntargets_true = []
    ntargets = []
    nhyps = []
    for k in range(35):
        # print()
        # print()
        print("k:", k)
        if k > 0:
            tracker.predict(1)
            for t in targets:
                t[0:2] += t[2:]

        clutter = [np.random.multivariate_normal(c[0], c[1])
                   for c in clutter_centroids for _ in range(c[2])]
        reports = {mht.Report(
            np.random.multivariate_normal(t[0:2], np.diag([0.1, 0.1])),  # noqa
            # t[0:2],
            np.eye(2) * 0.001,
            mht.models.position_measurement,
            i)
            for i, t in enumerate(targets)}
        false_reports = {mht.Report(
            np.random.multivariate_normal(t[0:2], np.diag([0.1, 0.1])),  # noqa
            np.eye(2) * 0.3,
            mht.models.position_measurement)
            for t in clutter}

        ntt = 0
        for s in sensors:
            sr = {r for r in reports if s.in_fov(r.z[0:2])}
            fsr = {r for r in false_reports if s.in_fov(r.z[0:2])}
            ntt += len(sr)
            reports -= sr
            this_scan = mht.Scan(s, list(sr | fsr))
            # mht.plot.plot_scan(this_scan)
            tracker.register_scan(this_scan)
        tracker._load_clusters()
        mlhyp = next(tracker.global_hypotheses())
        nclusters.append(len(tracker.active_clusters))
        ntargets.append(len(mlhyp.targets))
        ntargets_true.append(len(targets))
        nhyps.append(reduce(operator.mul, (len(c.hypotheses)
                                           for c in tracker.active_clusters)))
        plt.plot([t[0] for t in targets], [t[1] for t in targets],
                 marker='D', color='y', alpha=.5, linestyle='None')

    plt.plot([t[0] for t in targets], [t[1] for t in targets],
             marker='D', color='y', alpha=.5, linestyle='None')
    mht.plot.plot_hyptrace(mlhyp, covellipse=False)
    mht.plot.plot_hypothesis(mlhyp, cseed=2)
    mht.plot.plt.axis([-30, 150, -30, 150])
    mht.plot.plt.ylabel('Tracks')
    for s in sensors:
        mht.plot.plot_bbox(s)
    tracker._load_clusters()
    print("Clusters:", len(tracker.active_clusters))
    for c in tracker.active_clusters:
        mht.plot.plot_bbox(c)
    # mht.plot.plt.figure()
    # mht.plot.plt.subplot(3, 1, 1)
    # mht.plot.plt.plot(nclusters)
    # mht.plot.plt.axis([-1, k + 1, min(nclusters) - 0.1, max(nclusters) + 0.1])
    # mht.plot.plt.ylabel('# Clusters')
    # mht.plot.plt.subplot(3, 1, 2)
    # mht.plot.plt.plot(ntargets, label='Estimate')
    # mht.plot.plt.plot(ntargets_true, label='True')
    # mht.plot.plt.ylabel('# Targets')
    # mht.plot.plt.legend()
    # mht.plot.plt.axis([-1, k + 1, min(ntargets + ntargets_true) - 0.1,
                       # max(ntargets + ntargets_true) + 0.1])
    # mht.plot.plt.subplot(3, 1, 3)
    # mht.plot.plt.plot(nhyps)
    # mht.plot.plt.axis([-1, k + 1, min(nhyps) - 0.1, max(nhyps) + 0.1])
    # mht.plot.plt.ylabel('# Hyps')

    # for tr in mlhyp.tracks:
        # print(tr.trlen, len(set(tr.sources)), tr.sources)


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    # cProfile.run('draw()', sort='tottime')
    draw()
    if args.show:
        plt.show()
    else:
        plt.gcf().savefig(os.path.splitext(os.path.basename(__file__))[0],
                          bbox_inches='tight')


if __name__ == '__main__':
    main(*sys.argv[1:])
