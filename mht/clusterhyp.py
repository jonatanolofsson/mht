"""Implementation of the cluster-global hypothesis class."""

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


class ClusterHypothesis:
    """Class to represent a cluster hypothesis."""

    def __init__(self):
        """Init."""
        self.total_score = 0
        self.tracks = []

    @staticmethod
    def initial(tracks):
        """Create initial hypothesis."""
        self = ClusterHypothesis()
        self.tracks = tracks
        self.targets = {tr.target for tr in self.tracks}
        self.calculate_score()
        return self

    @staticmethod
    def new(phyp, assignments, sensor):
        """Create new hypothesis."""
        self = ClusterHypothesis()
        self.tracks = [track.assign(report, sensor)
                       for report, track in assignments]

        missed = set(phyp.tracks) - {tr for _, tr in assignments}
        self.tracks += [tr.missed(sensor)
                        for tr in missed
                        if tr.exist_score > 1]

        self.targets = {tr.target for tr in self.tracks}

        self.calculate_score()

        return self

    @staticmethod
    def merge(hyps):
        """Merge n hyps."""
        self = ClusterHypothesis()
        self.tracks = [tr for c in hyps for tr in c.tracks]
        self.targets = {tr.target for tr in self.tracks}
        self.calculate_score()
        return self

    def split(self, split_targets):
        """Return a subhypothesis to cover the provided targets."""
        tracks = [tr for tr in self.tracks if tr.target in split_targets]
        if len(tracks) == 0:
            return None
        h = ClusterHypothesis()
        h.tracks = tracks
        h.targets = {tr.target for tr in h.tracks}
        h.calculate_score()
        return h

    def calculate_score(self):
        """Calculate score."""
        self.total_score = sum(tr.score() for tr in self.tracks)

    def score(self):
        """Return the total score of the hypothesis."""
        return self.total_score

    def __eq__(self, b):
        """Check if self == b."""
        return self.tracks == b.tracks

    def __hash__(self):
        """Return hash."""
        return hash(tuple(self.tracks))

    def __gt__(self, b):
        """Check which hypothesis is better."""
        return self.score() > b.score()

    def __repr__(self):
        """Generate string representing the hypothesis."""
        return """::::: Hypothesis, score {} :::::
Tracks:
\t{}
    """.format(self.score(),
               "\n\t".join(
                   str(track)
                   for track in sorted(self.tracks, key=lambda x: x._id)))
