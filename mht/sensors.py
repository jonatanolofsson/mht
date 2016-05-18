"""File for sensor-related stuff."""


class EyeOfMordor:
    """Ideal sensor that sees all."""

    def __init__(self, score_false, score_new, score_miss):
        """Init."""
        self.score_false = score_false
        self.score_new = score_new
        self.score_miss = score_miss
