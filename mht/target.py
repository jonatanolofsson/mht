"""Methods to handle MHT target."""

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

from .track import Track


class Target:
    """Class to represent a single MHT target."""

    def __init__(self, cluster):
        """Init."""
        self._id = self.__class__._counter
        self.__class__._counter += 1
        self.cluster = cluster
        self.tracks = {}
        self.reset()

    @staticmethod
    def initial(cluster, filter):
        """Create initial target."""
        self = Target(cluster)
        self.tracks = {None: Track.initial(self, filter)}
        self.reset()
        return self

    @staticmethod
    def new(cluster, filter, report, sensor):
        """Create new target."""
        self = Target(cluster)
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
