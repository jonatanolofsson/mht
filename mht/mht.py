"""Library implementing Multiple Hypothesis assigning."""

from itertools import chain, product

from . import kf
from .cluster import Cluster
from .clusterhyp import ClusterHypothesis


class MHT:
    """MHT class."""

    def __init__(self, initial_targets=None, init_target_filter=None,
                 k_max=None):
        """Init."""
        self.init_target_filter = init_target_filter or kf.kfinit(0.1)
        self.clusters = {
            Cluster(self, initial_targets, init_target_filter, k_max)}
        self._split_clusters()

    def predict(self, dT):
        """Move to next timestep."""
        for cluster in self.clusters:
            cluster.predict(dT)

    def global_hypotheses(self):
        """Return global hypotheses."""
        # FIXME
        (c, ) = self.clusters
        return c.cluster_hypotheses

    def _match_clusters(self, m):
        """Select clusters within reasonable range."""
        return self.clusters

    def _split_clusters(self):
        """Split clusters."""
        pass

    def _cluster(self, scan):
        """Update clusters."""
        affected_clusters = set()

        def _merge_clusters(self, clusters):
            """Merge multiple clusters."""
            nonlocal affected_clusters
            c = Cluster(self,
                        initial_hypotheses=[ClusterHypothesis(
                            set().union(h.tracks for h in hyps),
                            set().union(h.parent_tracks for h in hyps),
                            set().union(h.score for h in hyps))
                            for hyps in product(*clusters)])
            c.assigned_reports = set.union(
                c.assigned_reports for c in clusters)
            affected_clusters -= clusters
            affected_clusters |= c
            self.clusters -= clusters
            self.clusters |= c

        for m in scan.reports:
            cmatches = set(self._match_clusters(m))
            affected_clusters |= cmatches

            if len(cmatches) > 1:
                cluster = self._merge_clusters(cmatches)
            elif len(cmatches) == 0:
                cluster = Cluster(self)
            else:
                (cluster,) = cmatches

            cluster.assigned_reports.add(m)

        for c in affected_clusters:
            yield (c, c.assigned_reports)
            c.assigned_reports = set()

    def register_scan(self, scan):
        """Register new scan."""
        for cluster, creports in self._cluster(scan):
            cluster.register_scan(Scan(scan.sensor, creports))

    def targets(self):
        """Retrieve all targets in filter."""
        yield from chain.from_iterable(c.targets for c in self.clusters)


class Report:
    """Class for containing reports."""

    def __init__(self, z, R, mfn):
        """Init."""
        self.z = z
        self.R = R
        self.mfn = mfn

    def __repr__(self):
        """Return string representation of reports."""
        return "R({}, R)".format(self.z.T)


class Scan:
    """Report container class."""

    def __init__(self, sensor, reports):
        """Init."""
        self.sensor = sensor
        self.reports = reports
