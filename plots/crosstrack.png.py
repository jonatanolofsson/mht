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
import matplotlib.pyplot as plt
# import cProfile

sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
import mht


np.random.seed(1)


def draw():
    """Create plot."""
    tracker = mht.MHT(
        cparams=mht.ClusterParameters(k_max=100, nll_limit=4, hp_limit=5),
        matching_algorithm="naive")
    targets = [
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 10.0, 1.0, -1.0]),
    ]
    hyps = None
    nclusters = []
    ntargets_true = []
    ntargets = []
    nhyps = []
    for k in range(25):
        if k > 0:
            tracker.predict(1)
            for t in targets:
                t[0:2] += t[2:]
        if k == 5:
            targets.append(np.array([5.0, 5.0, 1.0, 0.0]))
        if k % 7 == 0:
            targets.append(np.random.multivariate_normal(
                np.array([k, 7.0, 0.0, 0.0]),
                np.diag([0.5] * 4)))
        if k % 7 == 1:
            del targets[-1]
        if k == 10:
            targets.append(np.array([10.0, -30.0, 1.0, -0.5]))
        if k == 20:
            targets.append(np.array([k, 0.0, 1.0, 4.0]))

        reports = {mht.Report(
            np.random.multivariate_normal(t[0:2], np.diag([0.1, 0.1])),  # noqa
            # t[0:2],
            np.eye(2) * 0.001,
            mht.models.position_measurement,
            i)
            for i, t in enumerate(targets)}
        this_scan = mht.Scan(mht.sensors.EyeOfMordor(10, 3), reports)
        tracker.register_scan(this_scan)
        hyps = list(tracker.global_hypotheses())
        nclusters.append(len(tracker.active_clusters))
        ntargets.append(len(hyps[0].targets))
        ntargets_true.append(len(targets))
        nhyps.append(len(hyps))
        mht.plot.plot_scan(this_scan)
        plt.plot([t[0] for t in targets],
                 [t[1] for t in targets],
                 marker='D', color='y', alpha=.5, linestyle='None')
    mht.plot.plot_hyptrace(hyps[0], covellipse=True)
    mht.plot.plt.axis([-1, k + 1, -k - 1, k + 1 + 10])
    mht.plot.plt.ylabel('Tracks')
    mht.plot.plt.figure()
    mht.plot.plt.subplot(3, 1, 1)
    mht.plot.plt.plot(nclusters)
    mht.plot.plt.axis([-1, k + 1, min(nclusters) - 0.1, max(nclusters) + 0.1])
    mht.plot.plt.ylabel('# Clusters')
    mht.plot.plt.subplot(3, 1, 2)
    mht.plot.plt.plot(ntargets, label='Estimate')
    mht.plot.plt.plot(ntargets_true, label='True')
    mht.plot.plt.ylabel('# Targets')
    mht.plot.plt.legend()
    mht.plot.plt.axis([-1, k + 1, min(ntargets + ntargets_true) - 0.1,
                       max(ntargets + ntargets_true) + 0.1])
    mht.plot.plt.subplot(3, 1, 3)
    mht.plot.plt.plot(nhyps)
    mht.plot.plt.axis([-1, k + 1, min(nhyps) - 0.1, max(nhyps) + 0.1])
    mht.plot.plt.ylabel('# Hyps')


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
