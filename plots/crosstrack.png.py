"""Create crosstrack.png plot."""
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
        initial_targets=[
            mht.kf.KFilter(
                mht.models.ConstantVelocityModel(0.1),
                np.matrix([[0.0], [0.0], [1.0], [1.0]]),
                np.eye(4)
            ),
            mht.kf.KFilter(
                mht.models.ConstantVelocityModel(0.1),
                np.matrix([[0.0], [10.0], [1.0], [-1.0]]),
                np.eye(4)
            )
        ],
        cparams=mht.ClusterParameters(k_max=50, nll_limit=4, hp_limit=7)
        # , matching_algorithm="rtree"
        )
    targets = [
        np.array([[0.0], [0.0], [1.0], [1.0]]),
        np.array([[0.0], [10.0], [1.0], [-1.0]]),
    ]
    hyps = None
    nclusters = []
    ntargets_true = []
    ntargets = []
    nhyps = []
    for k in range(25):
        # print()
        # print()
        # print()
        # print()
        # print("k:", k)
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
        if k == 10:
            targets.append(np.array([[10.0], [-30.0], [1.0], [-0.5]]))
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
        hyps = list(tracker.global_hypotheses())
        # print("hp:", hyps[0].score(), hyps[-1].score())
        nclusters.append(len(tracker.active_clusters))
        ntargets.append(len(hyps[0].targets))
        ntargets_true.append(len(targets))
        nhyps.append(len(hyps))
        # mht.plot.plot_hypothesis(hyps[0], cseed=2)
        mht.plot.plot_scan(this_scan)
        plt.plot([t[0] for t in targets],
                 [t[1] for t in targets],
                 marker='D', color='y', alpha=.5, linestyle='None')
    print(hyps[0])
    mht.plot.plot_hyptrace(hyps[0], covellipse=False)
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
