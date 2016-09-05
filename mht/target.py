"""Methods to handle MHT target."""

from .track import Track


class Target:
    """Class to represent a single MHT target."""

    def __init__(self, tracker):
        """Init."""
        self._id = self.__class__._counter
        self.__class__._counter += 1
        self.tracker = tracker
        self.reset()

    @staticmethod
    def initial(tracker, filter):
        """Create initial target."""
        self = Target(tracker)
        self.tracks = {None: Track.initial(self, filter)}
        self.reset()
        return self

    @staticmethod
    def new(tracker, filter, report, sensor):
        """Create new target."""
        self = Target(tracker)
        tr = Track.new(self, filter, sensor, report)
        self.tracks = {report: tr}
        self.new_tracks[report] = tr
        return self

    def finalize_assignment(self, new_tracks):
        """Finalize assigment."""
        for tr in self.tracks.values():
            tr.children = {r: c for r, c in tr.children.items()
                           if c in new_tracks}
        self.tracks = {tr.report: tr for tr in new_tracks}
        self.reset()

    def predict(self, dT):
        """Move to next time step."""
        for track in self.tracks.values():
            track.predict(dT)

    def reset(self):
        """Reset caches etc."""
        self.new_tracks = {}

    def __repr__(self):
        """String representation of object."""
        return "T({})".format(self._id)
Target._counter = 0
