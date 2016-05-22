"""Methods to handle MHT target."""

from copy import deepcopy


TARGET_COUNTER = 0


class Target:
    """Class to represent a single MHT target."""

    def __init__(self, filter, score, report=None):
        """Init."""
        global TARGET_COUNTER
        self.cid = TARGET_COUNTER
        TARGET_COUNTER += 1
        self.reset()
        trck = Track(None,
                     None,
                     self,
                     filter=filter,
                     initial_score=score)

        self.tracks = [trck]
        if report is not None:
            self.new_tracks[report] = trck

    def assign(self, parent, m):
        """Assign measurement to track node to expand tree."""
        if m not in self.new_tracks:
            self.new_tracks[m] = Track(m, parent, self)
        return self.new_tracks[m]

    def finalize_assignment(self):
        """Finalize assigment."""
        self.tracks = self.new_tracks.values()
        self.reset()

    def predict(self, dT):
        """Move to next time step."""
        for track in self.tracks:
            track.predict(dT)

    def reset(self):
        """Reset caches etc."""
        self.new_tracks = {}
        self._score_cache = {}

    def score(self, report):
        """Return the score for a given report, for all tracks."""
        if report not in self._score_cache:
            self._score_cache[report] = [(track.score(report), track)
                                         for track in self.tracks]
        return self._score_cache[report]

    def __repr__(self):
        """String representation of object."""
        return "T({})".format(self.cid)


class Track:
    """Class to represent the tracks in a target tree."""

    def __init__(self, m, parent_track, target,
                 filter=None, initial_score=None):
        """Init."""
        self.parent_track_id = id(parent_track) if parent_track else id(target)
        self.parent_track = parent_track
        self.filter = filter or deepcopy(parent_track.filter)
        if m:
            self.my_score = self.filter.correct(m)
        else:
            self.my_score = initial_score
        self.target = target

    def is_new_target(self):
        """Check if target is brand new."""
        return id(self.target) == self.parent_track_id

    def assign(self, m):
        """Assign measurement to track."""
        return self.target.assign(self, m)

    def predict(self, dT):
        """Move to next time step."""
        self.filter.predict(dT)

    def score(self, m=None):
        """Find the score of assigning a measurement to the filter."""
        if m is None:
            return self.my_score
        return self.filter.score(m)

    def trace(self):
        """Get the trace of tracks that led to this."""
        tr = self.parent_track.trace() if self.parent_track else []
        return tr + [self]

    def __repr__(self):
        """Return string representation of object."""
        return "Tr(0x{:x}: {})".format(
            id(self),
            "[{}]".format(", ".join(str(int(x)) for x in self.filter.x)))

    def __lt__(self, b):
        """Check if self < b."""
        return id(self) < id(b)
