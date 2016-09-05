"""Create crosstrack.png plot."""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(
    os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
import mht


np.random.seed(1)


def draw():
    """Create plot."""
    mht.plot.plt.subplot(2, 1, 1)
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
        )
    ], k_max=2)
    targets = [
        np.array([[0.0], [0.0], [1.0], [1.0]]),
        np.array([[0.0], [10.0], [1.0], [-1.0]]),
    ]
    hyps = None
    nclusters = []
    for k in range(50):
        if k > 0:
            tracker.predict(1)
            for t in targets:
                t[0:2] += t[2:]
        if k == 5:
            targets.append(np.array([[5.0], [5.0], [1.0], [0.0]]))
        if k % 7 == 0:
            targets.append(np.array([[k], [7.0], [0.0], [0.0]])
                           + np.random.normal(size=(4, 1)) * 0.3)
        if k % 7 == 1:
            del targets[-1]
        if k == 20:
            targets.append(np.array([[k], [0.0], [1.0], [4.0]]))

        this_scan = mht.Scan(
            mht.sensors.EyeOfMordor(10, 3),
            [mht.Report(
                t[0:2] + np.random.normal(size=(2, 1)) * 0.3,
                np.eye(2),
                mht.models.position_measurement)
             for t in targets])
        tracker.register_scan(this_scan)
        nclusters.append(len(tracker.clusters))
        hyps = list(tracker.global_hypotheses())
        mht.plot.plot_hypothesis(hyps[0], cseed=2)
        mht.plot.plot_scan(this_scan)
        plt.plot([t[0] for t in targets],
                 [t[1] for t in targets],
                 marker='D', color='y', linestyle='None')
    print(hyps)
    mht.plot.plot_hyptrace(hyps[0], covellipse=False)
    mht.plot.plt.axis([-1, k + 1, -k - 1, k + 1 + 10])
    mht.plot.plt.subplot(2, 1, 2)
    mht.plot.plt.plot(nclusters)
    mht.plot.plt.axis([-1, k + 1, min(nclusters) - 0.1, max(nclusters) + 0.1])


def parse_args(*argv):
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--show', action="store_true")
    return parser.parse_args(argv)


def main(*argv):
    """Main."""
    args = parse_args(*argv)
    draw()
    if args.show:
        plt.show()
    else:
        plt.gcf().savefig(os.path.splitext(os.path.basename(__file__))[0],
                          bbox_inches='tight')


if __name__ == '__main__':
    main(*sys.argv[1:])
