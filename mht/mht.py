"""Library implementing Multiple Hypothesis Tracking."""

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

from itertools import chain
import sqlite3
import pickle
import multiprocessing as mp

from .cluster import Cluster, ClusterParameters
from .hypgen import permgen
from .utils import overlap, gaussian_bbox


def cluster_initer_factory(tracker, cparams):
    """Cluster initer factory."""
    def inner(self):
        self._id = tracker._new_cluster_id()
        self.params = cparams
    return inner


def predict_cluster(args):
    """Perform parallel time update on cluster."""
    (cluster, dT) = args
    cluster.predict(dT)
    return cluster


def correct_cluster(args):
    """Update cluster from multithread process."""
    (scan, cluster) = args
    # print('Updating cluster with {} targets.'.format(len(cluster.targets)))
    cluster.register_scan(scan)
    return cluster


class MHT:
    """MHT class."""

    def __init__(self, cparams=None, matching_algorithm=None,
                 dbfile=':memory:'):
        """Init."""
        self.matching_algorithm = matching_algorithm
        self.cparams = cparams if cparams else ClusterParameters()
        self.cluster_initer = cluster_initer_factory(self, self.cparams)

        self.active_clusters = set()

        self.dbfile = dbfile
        self.dbc = sqlite3.connect(dbfile)
        self.db = self.dbc.cursor()
        self._init_db()

        self.mppool = mp.Pool()

    def initiate_clusters(self, initial_targets):
        """Init clusters."""
        self._save_clusters({Cluster.initial(self.cluster_initer, [f])
                             for f in initial_targets})

    def _reboot(self):
        """Reboot filter."""
        self.dbc = sqlite3.connect(self.dbfile)
        self.db = self.dbc.cursor()
        self._init_db()
        self.active_clusters = set()

    def _init_db(self):
        """Init database."""
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS clusters ("
            "id     INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE,"
            "min_x  REAL,"
            "max_x  REAL,"
            "min_y  REAL,"
            "max_y  REAL,"
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
        self.db.execute("DELETE FROM clusters WHERE id IN ({});".format(ids))
        if self.matching_algorithm == "rtree":
            self.db.execute("DELETE FROM cluster_index WHERE id IN ({});"
                            .format(ids))
        self.dbc.commit()

    def _load_clusters(self, bbox=None):
        """Load clusters."""
        if self.matching_algorithm is None or bbox is None:
            self.active_clusters = self.query_clusters()
        elif self.matching_algorithm == "naive":
            all_clusters = self.query_clusters()
            self.active_clusters = {
                c for c in all_clusters
                if overlap(c.bbox(), bbox)}
        else:
            self.active_clusters = self.query_clusters(bbox)

    def query_clusters(self, bbox=None):
        """Get clusters intersecting boundingbox."""
        if bbox is None:
            pickles = self.db.execute("SELECT data FROM clusters")
        else:
            def get_clusters(bbox):
                if self.matching_algorithm == "db":
                    return self.db.execute((
                        "SELECT data FROM clusters WHERE "
                        "max_x >= {} AND "
                        "min_x <= {} AND "
                        "max_y >= {} AND "
                        "min_y <= {}"
                        ";").format(*bbox))
                elif self.matching_algorithm == "rtree":
                    #  PySQLite standard formatting doesn't work for some
                    #  reason.. bug? Using .format instead, since known data.
                    return self.db.execute((
                        "SELECT clusters.data FROM clusters "
                        "INNER JOIN cluster_index "
                        "ON clusters.id = cluster_index.id WHERE "
                        "cluster_index.max_x >= {} AND "
                        "cluster_index.min_x <= {} AND "
                        "cluster_index.max_y >= {} AND "
                        "cluster_index.min_y <= {}"
                        ";").format(*bbox))

            # FIXME: Use multiple queries if around wrapping-points!
            pickles = get_clusters(bbox)
        return {pickle.loads(p[0]) for p in pickles}

    def _save_clusters(self, clusters=None):
        """Store cluster data in database."""
        if clusters is None:
            clusters = self.active_clusters
        if self.matching_algorithm == "rtree":
            for c in clusters:
                self.db.execute(("REPLACE INTO cluster_index "
                                 "(id, min_x, max_x, min_y, max_y) "
                                 "VALUES ({}, {}, {}, {}, {});"
                                 ).format(c._id, *c.bbox()))
        for c in clusters:
            self.db.execute("UPDATE clusters SET "
                            "min_x=?, max_x=?, min_y=?, max_y=?, data=? "
                            "WHERE id=?", c.bbox() + (pickle.dumps(c), c._id))
        self.dbc.commit()

    def _overlapping_clusters(self, r):
        """Select clusters within reasonable range."""
        return {c for c in self.active_clusters
                if any(overlap(tr.bbox(), r.bbox())
                       for t in c.targets for tr in t.tracks.values())}

    def _split_clusters(self):
        """Split clusters."""
        new_clusters = set()
        old_clusters = set()
        for c in self.active_clusters:
            nc = c.split(self.cluster_initer)
            if len(nc) != 1:
                old_clusters.add(c)
            new_clusters |= nc
        self._delete_clusters(old_clusters)
        self.active_clusters = new_clusters

    def _merge_clusters(self, clusters):
        """Merge multiple clusters."""
        c = Cluster.merge(self.cluster_initer, clusters)
        c.assigned_reports = {r for c in clusters
                              for r in c.assigned_reports}
        self._delete_clusters(clusters)
        self._add_clusters({c})
        return c

    def _cluster(self, scan):
        """Update clusters."""
        new_clusters = set()
        self._load_clusters(scan.sensor.bbox())

        for r in scan.reports:
            cmatches = self._overlapping_clusters(r)

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
        self.active_clusters = set(
            self.mppool.map(predict_cluster,
                            ((c, dT) for c in self.active_clusters)))
        self._save_clusters()

    def register_scan(self, scan):
        """Register new scan."""
        self.active_clusters = set(
            self.mppool.map(
                correct_cluster,
                ((Scan(scan.sensor, cr), c)
                 for c, cr in self._cluster(scan))))
        self._split_clusters()
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

    def __init__(self, z, R, mfn, source=None, tpos=None):
        """Init."""
        self.z = z
        self.R = R
        self.mfn = mfn
        self.assigned_tracks = set()
        self.source = source
        self.tpos = tpos
        self._bbox = gaussian_bbox(self.z[0:2], self.R[0:2, 0:2], 2)

    def bbox(self):
        """Return report bbox."""
        return self._bbox

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
