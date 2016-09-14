"""Motion and measurement models."""

import numpy as np


class ConstantVelocityModel:
    """Constant velocity motion model."""

    def __init__(self, q):
        """Init."""
        self.q = q

    def __call__(self, xprev, Pprev, dT):
        """Step model."""
        x = xprev
        F = np.matrix([[1, 0, dT, 0],
                       [0, 1, 0, dT],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        Q = np.matrix([[dT ** 3 / 3, 0, dT ** 2 / 2, 0],
                       [0, dT ** 3 / 3, 0, dT ** 2 / 2],
                       [0, 0, dT, 0],
                       [0, 0, 0, dT]]) * self.q
        x = F * xprev
        P = F * Pprev * F.T + Q

        return (x, P)


def position_measurement(x):
    """Velocity measurement model."""
    H = np.matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0]])
    return (H * x, H)


def velocity_measurement(x):
    """Velocity measurement model."""
    H = np.matrix([[0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return (H * x, H)
