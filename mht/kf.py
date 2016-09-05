"""Kalman Filter implementation for MHT target."""

from math import log as ln
from math import sqrt
from math import pi
from numpy.linalg import det
import numpy as np

from . import models


def kfinit(q):
    """Default target initiator."""
    def inner(report):
        model = models.constant_velocity_2d(q)
        x0 = np.matrix([report.z[0], report.z[1], 0.0, 0.0]).T
        P0 = np.eye(4)
        return KFilter(model, x0, P0)
    return inner


def from_report(r):
    """Init KFilter from report."""
    return KFilter()


class KFilter:
    """Kalman-filter target."""

    def __init__(self, model, x0, P0):
        """Init."""
        self.model = model
        self.x = x0
        self.P = P0

    def __repr__(self):
        """Return string representation of measurement."""
        return "T({}, P)".format(self.x)

    def predict(self, dT):
        """Perform motion prediction."""
        self.x, self.P = self.model(self.x, self.P, dT)

    def correct(self, m):
        """Perform correction (measurement) update."""
        zhat, H = m.mfn(self.x)
        dz = m.z - zhat
        S = H * self.P * H.T + m.R
        K = self.P * H.T * S.I
        self.x += K * dz
        self.P -= K * H * self.P

        # FIXME: lambda_ex / PD according to Bar-Shalom 2007
        score = dz.T * S.I * dz / 2.0 + ln(2 * pi * sqrt(det(S)))
        return float(score)

    def nll(self, m):
        """Get the nll score of assigning a measurement to the filter."""
        zhat, H = m.mfn(self.x)
        dz = m.z - zhat
        S = H * self.P * H.T + m.R
        score = dz.T * S.I * dz / 2.0 + ln(2 * pi * sqrt(det(S)))
        return float(score)
