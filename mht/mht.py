"""Library implementing Multiple Hypothesis Tracking."""

from itertools import chain
import sqlite3
import pickle

from .cluster import Cluster, ClusterParameters
from .hypgen import permgen
from .utils import overlap, gaussian_bbox


def cluster_initer_factory(tracker, cparams):
    """Cluster initer factory."""
    def inner(self):
        self._id = tracker._new_cluster_id()
        self.params = cparams
    return inner


class MHT:
    """MHT class."""

    def __init__(self, initial_targets=None, cparams=None,
                 matching_algorithm=None, dbfile=':memory:'):
        """Init."""
        initial_targets = initial_targets or []
        self.matching_algorithm = matching_algorithm
        self.cparams = cparams if cparams else ClusterParameters()
        self.cluster_initer = cluster_initer_factory(self, self.cparams)

        self.active_clusters = set()

        self.dbc = sqlite3.connect(dbfile)
        self.db = self.dbc.cursor()
        self._init_db()

        if initial_targets:
            self._save_clusters({Cluster.initial(self.cluster_initer, [f])
                                 for f in initial_targets})

    def _init_db(self):
        """Init database."""
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS clusters ("
            "id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,"
            "data   BLOB"
            ");")
        if self.matching_algorithm == "rtree":
            self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS cluster_index"
                            " USING rtree(id, min_x, max_x, min_y, max_y);")

    def _new_cluster_id(self):
        """Insert new row in db and retrieve id."""
        self.db.execute("INSERT INTO clusters DEFAULT VALUES")
        return self.db.lastrowid

    def _add_clusters(self, clusters):
        """Add clusters."""
        self.active_clusters |= clusters

    def _delete_clusters(self, clusters):
        """Remove clusters."""
        self.active_clusters -= clusters
        ids = ', '.join(str(c._id) for c in clusters)
        self.db.execute("DELETE FROM clusters WHERE id IN ({})".format(ids))
        if self.matching_algorithm == "rtree":
            self.db.execute("DELETE FROM clusters_index WHERE id IN ({})"
                            .format(ids))

    def _load_clusters(self, bbox=None):
        """Load clusters."""
        if self.matching_algorithm is None:
            self.active_clusters = self.query_clusters(None)
        elif self.matching_algorithm == "naive":
            all_clusters = self.query_clusters(None)
            self.active_clusters = {
                c for c in all_clusters
                if overlap(c.bbox(), bbox())}
        elif self.matching_algorithm == "rtree":
            self.active_clusters = self.query_clusters(bbox)

    def query_clusters(self, bbox=None):
        """Get clusters intersecting boundingbox."""
        if bbox is None:
            pickles = self.db.execute("SELECT data FROM clusters")
        else:
            def get_clusters(bbox):
                return self.db.execute(
                    "SELECT cluster.data FROM clusters "
                    "INNER JOIN cluster_index "
                    "ON clusters.id = cluster_index.id WHERE "
                    "cluster_index.max_x >= ? AND "
                    "cluster_index.min_x <= ? AND "
                    "cluster_index.max_y >= ? AND "
                    "cluster_index.min_y <= ?"
                    ";", bbox)

            # FIXME: Use multiple queries if around wrapping-points!
            pickles = get_clusters(bbox)
        return {pickle.loads(p[0]) for p in pickles}

    def _save_clusters(self, clusters=None):
        """Store cluster data in database."""
        if clusters is None:
            clusters = self.active_clusters
        if self.matching_algorithm == "rtree":
            for c in clusters:
                self.db.execute("REPLACE INTO clusters_index "
                                "(min_x, max_x, min_y, max_y) "
                                "VALUES (?, ?, ?, ?)", c.bbox())
        for c in clusters:
            self.db.execute("UPDATE clusters SET data=? WHERE id=?",
                            (pickle.dumps(c), c._id))

    def _match_clusters(self, r):
        """Select clusters within reasonable range."""
        return {c for c in self.active_clusters
                if any(overlap(c.bbox(), r.bbox()))}

    def _split_clusters(self):
        """Split clusters."""
        new_clusters = set()
        old_clusters = set()
        for c in self.active_clusters:
            nc = c.split(self.cluster_initer)
            if len(nc) > 1:
                old_clusters.add(c)
            new_clusters |= nc
        self._delete_clusters(old_clusters)
        self.active_clusters = new_clusters

    def _merge_clusters(self, clusters):
        """Merge multiple clusters."""
        c = Cluster.merge(self.cluster_initer, clusters)
        self._delete_clusters(clusters)
        self._add_clusters({c})
        return c

    def _cluster(self, scan):
        """Update clusters."""
        new_clusters = set()
        self._load_clusters(scan.sensor.bbox())

        for r in scan.reports:
            cmatches = self._match_clusters(r)

            if len(cmatches) > 1:
                cluster = self._merge_clusters(cmatches)
            elif len(cmatches) == 0:
                cluster = Cluster.empty(self.cluster_initer)
                new_clusters.add(cluster)
            else:
                (cluster,) = cmatches

            cluster.assigned_reports.add(r)

        self.active_clusters |= new_clusters

        for c in self.active_clusters:
            a = c.assigned_reports
            c.assigned_reports = set()
            yield (c, a)

    def predict(self, dT, bbox=None):
        """Move to next timestep."""
        self._load_clusters(bbox)
        for cluster in self.active_clusters:
            cluster.predict(dT)
        self._save_clusters()

    def register_scan(self, scan):
        """Register new scan."""
        print()
        print("before:", len(self.active_clusters))
        for cluster, creports in self._cluster(scan):
            cluster.register_scan(Scan(scan.sensor, creports))
        self._split_clusters()
        print("after:", len(self.active_clusters))
        for c in self.active_clusters:
            print(len(c.hypotheses))
        self._save_clusters()

    def global_hypotheses(self, bbox=None):
        """Return global hypotheses."""
        self._load_clusters(bbox)
        yield from (GlobalHypothesis(hyps) for hyps in
                    permgen(((h.score(), h) for h in c.hypotheses)
                            for c in self.active_clusters))

    def targets(self):
        """Retrieve all targets in tracker."""
        yield from chain.from_iterable(c.targets for c in self.active_clusters)


class GlobalHypothesis:
    """Class to represent a global hypothesis."""

    def __init__(self, hypotheses):
        """Init."""
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

    def bbox(self, nstd=2):
        """Return report bbox."""
        # FIXME: Cache!!!
        return gaussian_bbox(self.z[0:2], self.R[0:2, 0:2], nstd)

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
