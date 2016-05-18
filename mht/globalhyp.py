"""Implementation of the global hypothesis class."""


def ghyphash(track_ids):
    """Generate global hypothesis hash from assignment."""
    return hash(tuple(sorted(track_ids)))


class GlobalHypothesis:
    """Class to represent a global hypothesis."""

    def __init__(self, tracker, targets, scan, hyp):
        """Init."""
        parent_tracks, self.unassigned = hyp
        self.tracker = tracker
        self.tracks = [track.assign(report) for report, track in parent_tracks]

        hyphash = ghyphash([id(track) for _, track in parent_tracks])
        ph = self.tracker.global_hypotheses.get(hyphash, None)
        self.parent_score = self.parent_hypothesis().score() if ph else 0

        self.n_missed = len(targets) - len(parent_tracks)
        self.n_new = sum(1 if track.is_new_target() else 0
                         for _, track in parent_tracks)

        # FIXME
        self.my_score = \
            scan.sensor.score_miss * self.n_missed + \
            scan.sensor.score_new * self.n_new

        self.total_score = \
            sum(track.score() for track in self.tracks) \
            + self.parent_score \
            + self.my_score

    def score(self):
        """Return the total score of the hypothesis."""
        return self.total_score

    def __hash__(self):
        """Return an indexable id representing the hypothesis."""
        return ghyphash({id(track) for track in self.tracks})

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
