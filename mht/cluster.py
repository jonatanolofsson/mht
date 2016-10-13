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
from .utils import PrioItem, connected_components, LARGE
from .kf import DefaultTargetInit


class ClusterParameters:
    """Cluster parmeters."""

    def __init__(self, **kwargs):
        """Init."""
        for name, value in kwargs.items():
            self.__dict__[name] = value

ClusterParameters.k_max = 100
ClusterParameters.hp_limit = LARGE
ClusterParameters.init_target_tracker = DefaultTargetInit(0.1, 0.1)


class Cluster:
    """MHT class."""

    def __init__(self, initer):
        """Init."""
        self.targets = []
        self.hypotheses = []
        self.ambiguous_tracks = []
        self.assigned_reports = set()
        self.params = None
        initer(self)
        if self.params is None:
            self.params = ClusterParameters()

    @staticmethod
    def initial(initer, initial_target_filters):
        """Create initial cluster."""
        self = Cluster(initer)
        for f in initial_target_filters:
            self.targets.append(Target.initial(self, f))
        self.hypotheses = [ClusterHypothesis.initial(
            [t.tracks[None] for t in self.targets])]
        return self

    @staticmethod
    def empty(initer):
        """Create empty cluster."""
        return Cluster.initial(initer, [])

    @staticmethod
    def merge(initer, clusters):
        """Merge multiple clusters."""
        self = Cluster(initer)

        # Hypotheses
        self.hypotheses = [ClusterHypothesis.merge(hyps[0])
                           for hyps in islice(permgen([[(h.score(), h)
                                                        for h in c.hypotheses]
                                                       for c in clusters],
                                                      True),
                                              0, self.params.k_max)]
        self.normalise()

        # Targets
        self.targets = list({t for h in self.hypotheses for t in h.targets})
        for t in self.targets:
            t.cluster = self

        # Ambiguous tracks
        self.ambiguous_tracks = [
            atrs for c in clusters for atrs in c.ambiguous_tracks]

        return self

    def _splitter(self, initer, split_targets):
        """Perform actual split."""
        cl = Cluster(initer)

        # Targets
        cl.targets = split_targets

        # Hypotheses
        cl.hypotheses = sorted(list(
            {hyp for hyp in (h.split(split_targets)
                             for h in self.hypotheses) if hyp}))
        cl.normalise()

        # Targets
        cl.targets = list({t for h in cl.hypotheses for t in h.targets})
        for t in self.targets:
            t.cluster = cl

        # Ambiguous tracks
        cl.ambiguous_tracks = [{tr for tr in atrs
                                if tr.target in split_targets}
                               for atrs in self.ambiguous_tracks]
        cl.ambiguous_tracks = [atrs for atrs in cl.ambiguous_tracks
                               if len(atrs) > 1]

        return cl

    def split(self, initer):
        """Split cluster into multiple independent clusters."""
        if len(self.hypotheses) == 0:
            return set()
        connections = defaultdict(set)
        for atrs in self.ambiguous_tracks:
            targets = {tr.target for tr in atrs}
            for target in targets:
                connections[target].update(targets)
        new_clusters = list(connected_components(connections))

        unassigned_targets = set(self.targets).difference(*new_clusters)
        new_clusters += [{t} for t in unassigned_targets]
        if len(new_clusters) > 1:
            return {self._splitter(initer, c) for c in new_clusters}
        else:
            return {self}

    def predict(self, dT):
        """Move to next timestep."""
        for target in self.targets:
            target.predict(dT)

    def normalise(self):
        """Normalise hypothesis scores."""
        if len(self.hypotheses):
            scores = [h.score() for h in self.hypotheses]
            min_score = min(scores)
            c = log(sum(exp(min_score - s) for s in scores)) - min_score
            for h in self.hypotheses:
                h.total_score += c

    def register_scan(self, scan):
        """Register scan."""
        def hlimit(g):
            """Limit hypothesis draws."""
            min_score = LARGE
            scores = []
            for ph, c, h in islice(g, None, self.params.k_max):
                scores.append(c)
                min_score = min(min_score, c)
                s = log(sum(exp(min_score - s) for s in scores)) - min_score
                if c + s > self.params.hp_limit:
                    return
                yield ph, h

        new_ts = {}

        # Generate new hyptheses
        self.hypotheses = list(sorted(list({
            ch for ch in
            (ClusterHypothesis.new(ph, hyp, scan.sensor)
             for ph, hyp in hlimit(self._assignment_hypotheses(scan, new_ts)))
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
                    self,
                    self.params.init_target_tracker(report),
                    report, scan.sensor)
            return new_targets[report].tracks[report]

        def get_murties(ph):
            """Get hypothesis generator for parent hypothesis."""
            M = len(scan.reports)
            N = len(ph.tracks)  # Nof targets in hypothesis

            miss_all_score = sum(tr.miss_score(scan.sensor)
                                 for tr in ph.tracks)

            if M == 0:
                return iter([(ph.score() + miss_all_score, iter([]))])

            '''
            Form C-matrix:  |1 2|
            1: MxN Cost of assigning measurent r to target c.
            2: Diagonal MxM cost of extraneous report (new or false).
            All other values LARGE to be avoided by Murty algorithm.
            '''
            C = np.empty((M, N + M))
            C.fill(LARGE)
            for i, tr in enumerate(ph.tracks):
                C[range(M), i] = [tr.match_score(r, scan.sensor)
                                  for r in scan.reports]
            C[range(M), range(N, N + M)] = scan.sensor.score_extraneous

            # Murty solution S: (cost, assignments)
            return ((ph.score() + S[0] + miss_all_score,
                     ((r, ph.tracks[a] if a < N else new_target_track(r))
                      for r, a in zip(scan.reports, S[1])))
                    for S in murty(C))

        murties = ((ph, get_murties(ph)) for ph in self.hypotheses)

        '''
        Algorithm description:
        If all parent hyps has entered the comparison, draw next
         parent hyp as this may be a candidate to switch to.
        While current parent hypothesis is cheaper, draw from it
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
                yield ph, a[0], r
                a = next(m, None)
            if a:
                Q.put(PrioItem(a[0], (a, ph, m)))

    def bbox(self):
        """Get minimal boundingbox."""
        # FIXME: Cache!!!
        bboxes = (tr.bbox()
                  for t in self.targets for tr in t.tracks.values())
        minbox = next(bboxes)
        for bbox in bboxes:
            minbox = (min(minbox[0], bbox[0]),
                      max(minbox[1], bbox[1]),
                      min(minbox[2], bbox[2]),
                      max(minbox[3], bbox[3]))
        return minbox
