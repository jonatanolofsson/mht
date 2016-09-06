"""Library implementing Multiple Hypothesis Tracking."""

from itertools import chain

from . import kf
from .cluster import Cluster
from .hypgen import permgen
from .utils import LARGE


class MHT:
    """MHT class."""

    def __init__(self, initial_targets=None, init_target_tracker=None,
                 k_max=None, nll_limit=LARGE, hp_limit=LARGE,
                 matching_algorithm=None):
        """Init."""
        initial_targets = initial_targets or []
        self.init_target_tracker = init_target_tracker or kf.kfinit(0.1)
        self.new_clusters = set()
        self.clusters = {Cluster.initial(self, [f]) for f in initial_targets}
        self.k_max = k_max
        self.nll_limit = nll_limit
        self.hp_limit = hp_limit
        self.matching_algorithm = matching_algorithm

    def predict(self, dT):
        """Move to next timestep."""
        for cluster in self.clusters:
            cluster.predict(dT)

    def global_hypotheses(self):
        """Return global hypotheses."""
        yield from (GlobalHypothesis(self, hyps) for hyps in
                    permgen(((h.score(), h) for h in c.hypotheses)
                            for c in self.clusters))

    def _match_clusters(self, r, sensor):
        """Select clusters within reasonable range."""
        if self.matching_algorithm is None:
            matching = self.clusters
        elif self.matching_algorithm == "naive":
            matching = {c for c in self.clusters
                        if any(sensor.in_fov(tr.filter.x)
                               for t in c.targets
                               for tr in t.tracks.values())}
        if len(matching) == 0:
            matching = {Cluster.empty(self)}
            self.new_clusters |= matching
        return matching

    def _split_clusters(self):
        """Split clusters."""
        new_clusters = set()
        for c in self.clusters:
            new_clusters |= c.split()
        self.clusters = new_clusters

    def _cluster(self, scan):
        """Update clusters."""
        affected_clusters = set()

        def _merge_clusters(clusters):
            """Merge multiple clusters."""
            nonlocal affected_clusters
            c = Cluster.merge(self, clusters)
            affected_clusters -= clusters
            affected_clusters.add(c)
            self.clusters -= clusters
            self.clusters.add(c)
            return c

        for r in scan.reports:
            cmatches = set(self._match_clusters(r, scan.sensor))
            affected_clusters |= cmatches

            if len(cmatches) > 1:
                cluster = _merge_clusters(cmatches)
            elif len(cmatches) == 0:
                cluster = Cluster.empty(self)
            else:
                (cluster,) = cmatches

            cluster.assigned_reports.add(r)

        self.clusters |= self.new_clusters
        self.new_clusters = set()

        for c in affected_clusters:
            yield (c, c.assigned_reports)
            c.assigned_reports = set()

    def register_scan(self, scan):
        """Register new scan."""
        for cluster, creports in self._cluster(scan):
            cluster.register_scan(Scan(scan.sensor, creports))
        self._split_clusters()

    def targets(self):
        """Retrieve all targets in tracker."""
        yield from chain.from_iterable(c.targets for c in self.clusters)


class GlobalHypothesis:
    """Class to represent a global hypothesis."""

    def __init__(self, tracker, hypotheses):
        """Init."""
        self.tracker = tracker
        self.cluster_hypotheses = hypotheses[0]
        self.tracks = [tr for h in self.cluster_hypotheses for tr in h.tracks]
        self.targets = {tr.target for tr in self.tracks}
        self.total_score = sum(tr.score() for tr in self.tracks)

    def score(self):
        """Return the total score of the hypothesis."""
        return self.total_score

    def __gt__(self, b):
        """Check which hypothesis is better."""
        return self.score() > b.score()

    def __repr__(self):
        """Generate string representing the hypothesis."""
        return """::::: Global Hypothesis, score {} :::::
Tracks:
\t{}
    """.format(self.score(),
               "\n\t".join(str(track) for track in self.tracks))


class Report:
    """Class for containing reports."""

    def __init__(self, z, R, mfn):
        """Init."""
        self.z = z
        self.R = R
        self.mfn = mfn
        self.assigned_tracks = set()

    def __repr__(self):
        """Return string representation of reports."""
        return "R({}, R)".format(self.z.T)


class Scan:
    """Report container class."""

    def __init__(self, sensor, reports):
        """Init."""
        self.sensor = sensor
        self.reports = reports

    def __repr__(self):
        """Return a string representation of the scan."""
        return "Scan: {}".format(str(self.reports))
