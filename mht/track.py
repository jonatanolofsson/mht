"""Track class."""

from copy import deepcopy
from math import exp, log

from .utils import LARGE, overlap, overlap_pa

NEW_EXIST_SCORE = 1
MAX_EXIST_SCORE = 4


class Track:
    """Class to represent the tracks in a target tree."""

    def __init__(self, target, parent, filter, report):
        """Init."""
        self.target = target
        self.target.new_tracks[report] = self
        self.parent_id = parent._id if parent else None
        self.filter = filter
        self.report = report
        self.my_score = 0
        self.children = {}
        self.parent_score = parent.score() if parent else 0
        self.exist_score = parent.exist_score if parent else 0
        self._trid = self.__class__._counter
        self._id = target._id
        self.__class__._counter += 1

        self.sources = deepcopy(parent.sources) if parent else set()
        if report:
            self.sources.add(report.source)
        self.trlen = (parent.trlen + 1) if parent else 1

    @staticmethod
    def initial(target, filter):
        """Create new track for target."""
        self = Track(target, None, filter, None)
        self.my_score = 0
        self.exist_score = MAX_EXIST_SCORE
        return self

    @staticmethod
    def new(target, filter, sensor, report):
        """Create new track for target."""
        self = Track(target, None, filter, report)
        report.assigned_tracks.add(self)
        self.my_score = sensor.score_extraneous
        self.exist_score = NEW_EXIST_SCORE
        return self

    @staticmethod
    def extend(parent, report, sensor):
        """Create child track."""
        filt = parent.target.cluster.params.init_target_tracker(report, parent)
        score = filt.correct(report)
        self = Track(parent.target, parent, filt, report)
        self.my_score = score - sensor.score_found
        self.exist_score = min(parent.exist_score + 1, MAX_EXIST_SCORE)
        return self

    def missed(self, sensor):
        """Missed detection track."""
        if None not in self.children:
            new = Track(self.target, self, deepcopy(self.filter), None)
            if sensor.in_fov(self.filter.x):
                new.my_score = self.miss_score(sensor)
                new.exist_score = max(self.exist_score - 1, 0)
            else:
                new.my_score = 0
                new.exist_score = self.exist_score
            self.children[None] = new
        return self.children[None]

    def assign(self, report, sensor):
        """Assign report to track."""
        if report not in self.target.new_tracks:
            new = Track.extend(self, report, sensor)
            report.assigned_tracks.add(new)
            self.target.new_tracks[report] = new
        if report not in self.children:
            self.children[report] = self.target.new_tracks[report]
        return self.children[report]

    def is_new(self):
        """Return true if target is new."""
        return (self.parent_id is None)

    def predict(self, dT):
        """Move to next time step."""
        self.filter.predict(dT)

    def score(self):
        """Return track score."""
        return self.parent_score + self.my_score

    def match_score(self, r, sensor):
        """Find the score of assigning a report to the track."""
        if overlap(self.filter.bbox(), sensor.bbox()):
            nll = self.filter.nll(r)
            if nll < self.target.cluster.params.nll_limit:
                return nll - self.found_score(sensor) \
                    - self.miss_score(sensor)
            return LARGE
        return LARGE

    def found_score(self, sensor):
        """Find the score of assigning any report to the track."""
        score_miss = self.miss_score(sensor)
        return -log(1 - exp(-score_miss)) \
            if score_miss > 1e-8 else LARGE

    def miss_score(self, sensor):
        """Find the score of not assigning any report to the track."""
        return sensor.score_miss \
            * overlap_pa(self.filter.bbox(), sensor.bbox())

    def __repr__(self):
        """Return string representation of object."""
        return "Tr({}/{}/{}: {} {} {})".format(
            self._id,
            self._trid,
            self.parent_id if self.parent_id else "x",
            "[{}]".format(", ".join("{:.12f}".format(float(x))
                                    for x in self.filter.x)),
            "x" if self.report is None else self.report.source,
            self.exist_score
            )

    def __lt__(self, b):
        """Check if self < b."""
        return id(self) < id(b)
Track._counter = 0
