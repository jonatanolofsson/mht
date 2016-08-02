"""MHT Cluster."""
import queue
import numpy as np
# import matplotlib.pyplot as plt
# from . import plot

from .target import Target
from .clusterhyp import ClusterHypothesis
from .hypgen import permgen
from .hypgen import murty
from .utils import PrioItem

LARGE = 10000


class Cluster:
    """MHT class."""

    def __init__(self, filt, initial_targets=None, k_max=None,
                 initial_hypotheses=None):
        """Init."""
        self.filter = filt
        self.assigned_reports = set()
        self.targets = initial_targets if initial_targets else []
        if initial_targets:
            for target in self.targets:
                for track in target.tracks:
                    track.is_new_target = lambda: False

        self.k_max = k_max
        if initial_hypotheses is None:
            ch = ClusterHypothesis(self, self.targets, None, None)
            self.cluster_hypotheses = {ch: ch}
            self.best_hyp = ch
        else:
            self.cluster_hypotheses = {h: h for h in initial_hypotheses}
            self.best_hyp = min(self.cluster_hypotheses.values(),
                                key=lambda x: x.score())

    def predict(self, dT):
        """Move to next timestep."""
        for target in self.targets:
            target.predict(dT)

    def register_scan(self, scan):
        """Register scan."""
        new_hypotheses = {}
        self.best_hyp = None
        k = 0
        for hyp in self._hypothesis_factory(self.targets, scan):
            ch = ClusterHypothesis(self, self.targets, hyp, scan)
            new_hypotheses[ch] = ch
            self.best_hyp = ch if self.best_hyp is None else \
                min(self.best_hyp, ch)
            k += 1
            if self.k_max is not None and k >= self.k_max:
                break
        self.cluster_hypotheses = new_hypotheses
        for target in self.targets:
            target.finalize_assignment()

        # Delete targets with no tracks.
        self.targets = [target for target in self.targets
                        if len(target.tracks) > 0]

    def mlhyp(self):
        """Get most likely hypothesis."""
        return self.best_hyp

    def _hypothesis_factory(self, targets, scan):
        """Generate cluster hypotheses."""
        new_target_reports = {}

        def new_target(report):
            if report not in new_target_reports:
                obj = new_target_reports[report] = Target(
                    self.filter.init_target_filter(report),
                    score=scan.sensor.score_new,
                    report=report)
                self.targets.append(obj)
            return new_target_reports[report]

        def get_permgen(scan, tH):
            target_assignment = ((
                report,
                targets[a] if a < N else
                new_target(report) if a < N + M else
                None)
                for report, a in zip(scan.reports, tH[1]))

            return permgen((target.score(report) if target else [(0, None)] for
                            report, target in target_assignment))

        M = len(scan.reports)
        N = len(targets)
        C = np.empty((M, N + 2*M))
        C.fill(LARGE)
        for t, target in enumerate(targets):
            C[:, t] = [min(x[0] for x in target.score(r))
                       for r in scan.reports]
        C[range(M), range(N, N + M)] = scan.sensor.score_new
        C[range(M), range(N + M, N + 2*M)] = scan.sensor.score_false
        target_hypgen = murty(C)

        Q = queue.PriorityQueue()
        nxt_tH = next(target_hypgen)
        Q.put(PrioItem(nxt_tH[0], get_permgen(scan, nxt_tH)))
        nxt_tH = next(target_hypgen, None)
        while not Q.empty():
            pgen = Q.get_nowait().data
            next_break = min([x for x in [
                nxt_tH[0] if nxt_tH else None,
                Q.queue[0].prio if not Q.empty() else None,
                ] if x is not None] + [LARGE])
            for track_assignment, next_trackcost in pgen:
                yield list(zip(scan.reports, track_assignment))
                if next_trackcost and next_trackcost > next_break:
                    Q.put(PrioItem(next_trackcost, pgen))
                    break

            if nxt_tH and (Q.empty() or Q.queue[0].prio > nxt_tH[0]):
                Q.put(PrioItem(nxt_tH[0], get_permgen(scan, nxt_tH)))
                nxt_tH = next(target_hypgen, None)
