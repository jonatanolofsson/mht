"""Implementation of the cluster-global hypothesis class."""


class ClusterHypothesis:
    """Class to represent a cluster hypothesis."""

    def __init__(self, cluster):
        """Init."""
        self.cluster = cluster
        self.total_score = 0
        self.tracks = []

    @staticmethod
    def initial(cluster, tracks):
        """Create initial hypothesis."""
        self = ClusterHypothesis(cluster)
        self.tracks = tracks
        self.targets = {tr.target for tr in self.tracks}
        self.calculate_score()
        return self

    @staticmethod
    def new(cluster, phyp, assignments, sensor):
        """Create new hypothesis."""
        self = ClusterHypothesis(cluster)
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
    def merge(cluster, hyps):
        """Merge n hyps."""
        self = ClusterHypothesis(cluster)
        self.tracks = [x for c in hyps for x in c.tracks]
        self.targets = {tr.target for tr in self.tracks}
        self.calculate_score()
        return self

    def split(self, split_targets):
        """Return a subhypothesis to cover the provided targets."""
        tracks = [tr for tr in self.tracks if tr.target in split_targets]
        if len(tracks) == 0:
            return None
        h = ClusterHypothesis(self.cluster)
        h.tracks = tracks
        h.targets = {tr.target for tr in h.tracks}
        h.calculate_score()
        return h

    def calculate_score(self):
        """Calculate score."""
        # FIXME: Cache
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
               "\n\t".join(str(track) for track in self.tracks))
