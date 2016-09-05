"""File for sensor-related stuff."""

from math import exp, log


class EyeOfMordor:
    """Ideal sensor that sees all."""

    def __init__(self, score_extraneous, score_miss):
        """Init."""
        self.score_extraneous = score_extraneous
        self.score_miss = score_miss
        self.score_found = -log(1 - exp(-score_miss))
