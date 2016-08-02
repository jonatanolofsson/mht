"""Implementation of the cluster-global hypothesis class."""


def chyphash(track_ids):
    """Generate cluster hypothesis hash from assignment."""
    return hash(tuple(sorted(track_ids)))


class ClusterHypothesis:
    """Class to represent a cluster hypothesis."""

    def __init__(self, cluster, targets, parent_tracks, scan=None, score=None):
        """Init."""
        self.cluster = cluster
        self.targets = targets

        if parent_tracks is None:
            self.unassigned = []
            self.tracks = [target.tracks[0] for target in targets]
            self.parent_score = 0
            self.my_score = 0
            self.total_score = 0

            self.n_missed = 0
            self.n_new = len(targets)
            self.n_false = 0

        else:
            self.unassigned = [
                report for report, track in parent_tracks if track is None]
            self.tracks = [track.assign(report)
                           for report, track in parent_tracks
                           if track is not None]

            parent_hash = chyphash([id(track) for _, track in parent_tracks
                                    if track is not None])
            ph = self.cluster.cluster_hypotheses.get(parent_hash, None)
            self.parent_score = ph.score() if ph else 0

            self.n_missed = len(targets) - len(parent_tracks)
            self.n_new = sum(1 if track.is_new_target() else 0
                             for _, track in parent_tracks
                             if track is not None)
            self.n_false = len(self.unassigned)

            self.track_score = sum(track.score() for track in self.tracks)

            if score:
                self.total_score = score
            else:
                # FIXME
                my_score = self.track_score + \
                    scan.sensor.score_false * self.n_false + \
                    scan.sensor.score_miss * self.n_missed + \
                    scan.sensor.score_new * self.n_new

                self.total_score = self.parent_score + my_score

    def score(self):
        """Return the total score of the hypothesis."""
        return self.total_score

    def __hash__(self):
        """Return an indexable id representing the hypothesis."""
        return chyphash({id(track) for track in self.tracks})

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
