"""Methods to handle MHT target."""

from copy import copy


class Target:
    """Class to represent a single MHT target."""

    def __init__(self, filter):
        """Init."""
        self.tracks = {Track(None, None, self, filter=filter)}
        self.reset()

    def assign(self, parent, m):
        """Assign measurement to track node to expand tree."""
        if m not in self.new_tracks:
            self.new_tracks[m] = Track(m, parent, self)

    def predict(self, dT):
        """Move to next time step."""
        self.tracks = self.new_tracks
        for track in self.tracks:
            track.predict(dT)
        self.reset()

    def reset(self):
        """Reset caches etc."""
        self.new_tracks = {}
        self.score_cache = {}

    def score(self, report):
        """Return the score for a given report, for all tracks."""
        if report not in self.score_cache:
            self.score_cache[report] = {(l.score(report), l)
                                        for l in self.tracks}
        return self.score_cache[report]


class Track:
    """Class to represent the tracks in a target tree."""

    def __init__(self, m, parent_track, target, filter=None):
        """Init."""
        self.m = m
        self.filter = filter or copy(parent_track.filter)
        if m:
            self.filter.correct(m)
        self.target = target

    def assign(self, m):
        """Assign measurement to track."""
        self.target.assign(self, m)

    def predict(self, dT):
        """Move to next time step."""
        self.filter.predict(dT)

    def score(self, m):
        """Find the score of assigning a measurement to the filter."""
        return self.filter.score(m)
