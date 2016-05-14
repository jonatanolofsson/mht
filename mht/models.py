"""Motion and measurement models."""

import numpy as np


def constant_velocity_2d(q):
    """Constant velocity motion model."""
    def mdl(xprev, Pprev, dT):
        x = xprev
        F = np.matrix([[1, 0, dT, 0],
                       [0, 1, 0, dT],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        Q = np.matrix([[dT ** 3 / 3, 0, dT ** 2 / 2, 0],
                       [0, dT ** 3 / 3, 0, dT ** 2 / 2],
                       [0, 0, dT, 0],
                       [0, 0, 0, dT]]) * q
        x = F * xprev
        P = F * Pprev * F.T + Q

        return (x, P)
    return mdl


def velocity_measurement(x):
    """Velocity measurement model."""
    H = np.matrix([[0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return (H * x, H)
