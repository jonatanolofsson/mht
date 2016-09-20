"""File for sensor-related stuff."""

from math import exp, log

from .utils import LARGE, within


class Sensor:
    """Sensor base class."""

    def __init__(self, score_extraneous, score_miss):
        """Init."""
        self.score_extraneous = score_extraneous
        self.score_miss = score_miss
        self.score_found = -log(1 - exp(-score_miss)) \
            if score_miss > 0 else LARGE

        self._id = Sensor._counter
        Sensor._counter += 1
Sensor._counter = 0


class EyeOfMordor(Sensor):
    """Ideal sensor that sees all."""

    def __init__(self, score_extraneous, score_miss):
        """Init."""
        super(EyeOfMordor, self).__init__(score_extraneous, score_miss)

    def bbox(self):
        """Return FOV bbox."""
        return (-LARGE, LARGE, -LARGE, LARGE)

    def in_fov(self, state):
        """Return nll prob of detection, given fov."""
        return True


class Satellite(Sensor):
    """Satellite sensor with field-of-view."""

    def __init__(self, fov, score_extraneous, score_miss):
        """Init."""
        super(Satellite, self).__init__(score_extraneous, score_miss)
        self.fov = fov

    def bbox(self):
        """Return FOV bbox."""
        return self.fov

    def in_fov(self, state):
        """Return nll prob of detection, given fov."""
        return within(state, self.fov)
