"""Library implementing Multiple Hypothesis assigning."""

import queue
import numpy as np
from copy import copy
import matplotlib.pyplot as plt

from .target import Target
from . import kf
from . import models
from . import plot
from .globalhyp import GlobalHypothesis
from .hypgen import permgen
from .hypgen import murty

LARGE = 10000


def _init_target_filter(report):
    """Default target initiator."""
    model = models.constant_velocity_2d(0.1)
    x0 = np.matrix([report.z[0], report.z[1], 0.0, 0.0]).T
    P0 = np.eye(4)
    return kf.KFilter(model, x0, P0)


class MHT:
    """MHT class."""

    def __init__(self, initial_targets=None, init_target_filter=None):
        """Init."""
        self.init_target_filter = init_target_filter or _init_target_filter
        self.targets = initial_targets if initial_targets else []
        if initial_targets:
            for target in self.targets:
                for track in target.tracks:
                    track.is_new_target = lambda: False

        gh = GlobalHypothesis(self, initial_targets, None, None)
        self.global_hypotheses = {gh: gh}

    def predict(self, dT):
        """Move to next timestep."""
        for target in self.targets:
            target.predict(dT)

    def _relevant_targets(self, scan):
        """Filter out relevant targets."""
        return copy(self.targets)

    def register_scan(self, scan):
        """Register scan."""
        targets = self._relevant_targets(scan)
        new_hypotheses = {}
        best_hyp = None
        k = 0
        for hyp in self._hypothesis_factory(targets, scan):
            gh = GlobalHypothesis(self, targets, scan, hyp)
            new_hypotheses[gh] = gh
            best_hyp = gh if best_hyp is None else min(best_hyp, gh)
            k += 1
            plot.plot_hypothesis(gh)
            plt.axis([-1, 11, -1, 11])
            plt.show()
        self.global_hypotheses = new_hypotheses
        for target in self.targets:
            target.finalize_assignment()

        # Delete targets with no track.
        self.targets = [target for target in self.targets
                        if len(target.tracks) > 0]

    def _hypothesis_factory(self, targets, scan):
        """Generate global hypotheses."""
        new_target_reports = {}

        def new_target(report):
            if report not in new_target_reports:
                obj = new_target_reports[report] = Target(
                    self.init_target_filter(report),
                    score=scan.sensor.score_new)
                self.targets.append(obj)
            return new_target_reports[report]

        def get_permgen(scan, tH):
            target_assignment = ((
                report,
                targets[a] if a < N else
                new_target(report) if a < 2*N else
                None)
                for report, a in zip(scan.reports, tH[1]))

            return permgen((target.score(report) if target else [(0, None)] for
                            report, target in target_assignment))

        M = len(scan.reports)
        N = len(targets)
        C = np.empty((M, 2*N + M))
        C.fill(LARGE)
        for t, target in enumerate(targets):
            C[:, t] = [min(x[0] for x in target.score(r))
                       for r in scan.reports]
        C[range(N), range(N, 2*N)] = scan.sensor.score_new
        C[range(N), range(2*N, 2*N + M)] = scan.sensor.score_false
        target_hypgen = murty(C)

        Q = queue.PriorityQueue()
        nxt_tH = next(target_hypgen)
        Q.put((nxt_tH[0], get_permgen(scan, nxt_tH)))
        nxt_tH = next(target_hypgen, None)
        while not Q.empty():
            pgen= Q.get_nowait()[1]
            next_break = min([x[0] for x in [
                nxt_tH,
                Q.queue[0] if not Q.empty() else None,
                ] if x] + [LARGE])
            for track_assignment, next_trackcost in pgen:
                yield list(zip(scan.reports, track_assignment))
                if next_trackcost and next_trackcost > next_break:
                    Q.put((next_trackcost, pgen))
                    break

            if nxt_tH and (Q.empty() or Q.queue[0][0] > nxt_tH[0]):
                Q.put((nxt_tH[0], get_permgen(scan, nxt_tH)))
                nxt_tH = next(target_hypgen, None)


class Report:
    """Class for containing measurement."""

    def __init__(self, z, R, mfn):
        """Init."""
        self.z = z
        self.R = R
        self.mfn = mfn

    def __repr__(self):
        """Return string representation of measurement."""
        return "R({}, R)".format(self.z.T)


class Scan:
    """Report container class."""

    def __init__(self, sensor, reports):
        """Init."""
        self.sensor = sensor
        self.reports = reports
