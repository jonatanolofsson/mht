"""File for sensor-related stuff."""

from math import exp, log

from .utils import LARGE


class EyeOfMordor:
    """Ideal sensor that sees all."""

    def __init__(self, score_extraneous, score_miss):
        """Init."""
        self.score_extraneous = score_extraneous
        self.score_miss = score_miss
        self.score_found = -log(1 - exp(-score_miss)) \
            if score_miss > 0 else LARGE

    def in_fov(self, state):
        """Return nll prob of detection, given fov."""
        return True
