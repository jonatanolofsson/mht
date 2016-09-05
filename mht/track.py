"""Track class."""

from copy import deepcopy

NEW_EXIST_SCORE = 1
MAX_EXIST_SCORE = 4


class Track:
    """Class to represent the tracks in a target tree."""

    def __init__(self, target, parent, filter, report):
        """Init."""
        self.target = target
        self.target.new_tracks[report] = self
        self.parent = parent
        self.filter = filter
        self.report = report
        self.my_score = 0
        self.children = {}
        self.parent_score = parent.score() if parent else 0
        self.exist_score = parent.exist_score if parent else 0
        self._id = self.__class__._counter
        self.__class__._counter += 1

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
        filt = deepcopy(parent.filter)
        score = filt.correct(report)
        self = Track(parent.target, parent, filt, report)
        self.my_score = score - sensor.score_found
        self.exist_score = min(parent.exist_score + 1, MAX_EXIST_SCORE)
        return self

    def missed(self, sensor):
        """Missed detection track."""
        if None not in self.children:
            new = Track(self.target, self, deepcopy(self.filter), None)
            new.my_score = sensor.score_miss
            new.exist_score = max(self.exist_score - 1, 0)
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

    def predict(self, dT):
        """Move to next time step."""
        self.filter.predict(dT)

    def score(self):
        """Return track score."""
        return self.parent_score + self.my_score

    def match(self, r):
        """Find the score of assigning a report to the filter."""
        return self.filter.nll(r)

    def trace(self):
        """Backtrace lineage until end or any of the previously searched."""
        t = self
        while t:
            yield t
            t = t.parent

    def __repr__(self):
        """Return string representation of object."""
        return "Tr({}/{}/{}: {} {}{})".format(
            self.target._id,
            self._id,
            self.parent._id if self.parent else "x",
            "[{}]".format(", ".join("{:.12f}".format(float(x))
                                    for x in self.filter.x)),
            "x" if self.report is None else "",
            self.exist_score
            )

    def __lt__(self, b):
        """Check if self < b."""
        return id(self) < id(b)
Track._counter = 0
