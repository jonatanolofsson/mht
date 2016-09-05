"""MHT Cluster."""
import queue
from math import log, exp
import numpy as np
from itertools import islice
from collections import defaultdict
# import matplotlib.pyplot as plt
# from . import plot

from .target import Target
from .clusterhyp import ClusterHypothesis
from .hypgen import murty, permgen
from .utils import PrioItem, connected_components

LARGE = 10000


class Cluster:
    """MHT class."""

    def __init__(self, tracker):
        """Init."""
        self.tracker = tracker
        self.targets = []
        self.hypotheses = []
        self.ambiguous_tracks = []
        self.assigned_reports = set()

    @staticmethod
    def initial(tracker, initial_target_filters):
        """Create initial cluster."""
        self = Cluster(tracker)
        for f in initial_target_filters:
            self.targets.append(Target.initial(f))
        self.hypotheses = [ClusterHypothesis.initial(
            self, [t.tracks[None] for t in self.targets])]
        return self

    @staticmethod
    def empty(tracker):
        """Create empty cluster."""
        return Cluster.initial(tracker, [])

    @staticmethod
    def merge(tracker, clusters):
        """Merge multiple clusters."""
        self = Cluster(tracker)

        # Hypotheses
        self.hypotheses = [ClusterHypothesis.merge(self, hyps[0])
                           for hyps in islice(permgen([[(h.score(), h)
                                                        for h in c.hypotheses]
                                                       for c in clusters],
                                                      True),
                                              0, self.tracker.k_max)]
        self.normalise()

        # Targets
        self.targets = list({t for h in self.hypotheses for t in h.targets})

        # Ambiguous tracks
        self.ambiguous_tracks = [
            tr for c in clusters for tr in c.ambiguous_tracks]

        return self

    def _splitter(self, split_targets):
        """Perform actual split."""
        cl = Cluster(self.tracker)

        # Targets
        cl.targets = split_targets

        # Hypotheses
        cl.hypotheses = sorted(list(
            {hyp for hyp in (h.split(split_targets)
                             for h in self.hypotheses) if hyp}))
        cl.normalise()

        # Targets
        self.targets = list({t for h in self.hypotheses for t in h.targets})

        # Ambiguous tracks
        cl.ambiguous_tracks = [{tr for tr in atrs
                                if tr.target in split_targets}
                               for atrs in self.ambiguous_tracks]
        cl.ambiguous_tracks = [atrs for atrs in cl.ambiguous_tracks
                               if len(atrs) > 1]

        return cl

    def split(self):
        """Split cluster into multiple independent clusters."""
        connections = defaultdict(set)
        for atrs in self.ambiguous_tracks:
            targets = {tr.target for tr in atrs}
            for target in targets:
                connections[target].update(targets)
        new_clusters = list(connected_components(connections))

        unassigned_targets = set(self.targets).difference(*new_clusters)
        new_clusters += [{t} for t in unassigned_targets]
        if len(new_clusters) > 1:
            return {self._splitter(c) for c in new_clusters}
        else:
            return {self}

    def predict(self, dT):
        """Move to next timestep."""
        for target in self.targets:
            target.predict(dT)

    def normalise(self):
        """Normalise hypothesis scores."""
        if len(self.hypotheses):
            c = log(sum(exp(-h.score()) for h in self.hypotheses))
            for h in self.hypotheses:
                h.total_score += c

    def register_scan(self, scan):
        """Register scan."""
        new_ts = {}

        # Generate new hyptheses
        self.hypotheses = list(sorted(list({
            ch for ch in
            (ClusterHypothesis.new(self, ph, hyp, scan.sensor)
             for ph, hyp in islice(self._assignment_hypotheses(scan, new_ts),
                                   None, self.tracker.k_max))
            if len(ch.tracks) > 0})))
        self.normalise()

        # Handle created targets and assignments
        self.targets = list({t for h in self.hypotheses for t in h.targets})
        tracks = {tr for h in self.hypotheses for tr in h.tracks}
        for target in self.targets:
            target.finalize_assignment({tr for tr in tracks
                                        if tr.target is target})

        # Find tracks from reports that were assigned to multiple targets
        self.ambiguous_tracks = [
            set().union(*(tr.children.values() for tr in atrs)) & tracks
            for atrs in self.ambiguous_tracks]
        self.ambiguous_tracks = [atrs for atrs in self.ambiguous_tracks
                                 if len(atrs) > 1]
        for r in scan.reports:
            if len(r.assigned_tracks) > 1:
                self.ambiguous_tracks.append(r.assigned_tracks)

    def _assignment_hypotheses(self, scan, new_targets):
        """Generate cluster hypotheses."""
        def new_target_track(report):
            """Create new target."""
            '''
            The cache is global, as the new target is
            independent of the parent hypothesis.
            '''
            nonlocal new_targets
            if report not in new_targets:
                new_targets[report] = Target.new(
                    self.tracker.init_target_tracker(report),
                    report, scan.sensor)
            return new_targets[report].tracks[report]

        def get_murties(ph):
            """Get hypothesis generator for parent hypothesis."""
            M = len(scan.reports)
            N = len(ph.tracks)  # Nof targets in hypothesis

            def uniqify(g):
                """Dont return same value twice in a row from generator."""
                '''
                Compare only report assignments, as the lower right corner of C
                will produce non-unique assignments.
                '''
                last = None
                for i in g:
                    a = i[1][:M]
                    if a == last:
                        continue
                    last = a
                    yield i

            '''
            Form C-matrix:  |1 2|
                            |3 4|
            1: MxN Cost of assigning measurent r to target c.
            2: Diagonal MxM cost of extraneous report (new or false).
            3: Diagonal NxN cost of not assigning any report to target.
            4: NxM matrix filled with "cost" of assigning report to target.
            All other values LARGE to be avoided by Murty algorithm.
            '''
            C = np.empty((N + M, N + M))
            C.fill(LARGE)
            for i, tr in enumerate(ph.tracks):
                C[range(M), i] = [tr.match(r) - scan.sensor.score_found
                                  for r in scan.reports]
            C[range(M), range(N, N + M)] = scan.sensor.score_extraneous
            C[range(M, N + M), range(N)] = scan.sensor.score_miss
            C[M:N + M, N:N + M] = 0

            # Murty solution S: (cost, assignments)
            return ((ph.score() + S[0],
                     ((r, ph.tracks[a] if a < N else new_target_track(r))
                      for r, a in zip(scan.reports, S[1])))
                    for S in uniqify(murty(C)))

        murties = ((ph, get_murties(ph)) for ph in self.hypotheses)

        '''
        Algorithm description:
        If all parent hyps has entered the comparison, draw next
         parent hyp as this may be a candidate to switch to.
        While current parent hypothesis is cheaper, draw from its
         children assignments.
        '''
        Q = queue.PriorityQueue()
        ph, m = next(murties)
        a = next(m)
        last_item = PrioItem(a[0], (a, ph, m))
        Q.put(last_item)
        while not Q.empty():
            item = Q.get_nowait()
            if item == last_item:
                ph, m = next(murties, (None, None))
                if m:
                    a = next(m)
                    last_item = PrioItem(a[0], (a, ph, m))
                    Q.put(last_item)
            a, ph, m = item.data
            next_break = Q.queue[0].prio if not Q.empty() else LARGE
            while a and a[0] <= next_break:
                r = list(a[1])
                yield ph, r
                a = next(m, None)
            if a:
                Q.put(PrioItem(a[0], (a, ph, m)))
