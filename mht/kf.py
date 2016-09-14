"""Kalman Filter implementation for MHT target."""

from math import log as ln
from math import sqrt, pi
from numpy.linalg import det
import numpy as np

from . import models
from .utils import gaussian_bbox


class DefaultTargetInit:
    """Default target initiator."""

    def __init__(self, q):
        """Init."""
        self.q = q

    def __call__(self, report):
        """Init new target from report."""
        model = models.ConstantVelocityModel(self.q)
        x0 = np.matrix([report.z[0], report.z[1], 0.0, 0.0]).T
        P0 = np.eye(4) * 2
        return KFilter(model, x0, P0)


class KFilter:
    """Kalman-filter target."""

    def __init__(self, model, x0, P0):
        """Init."""
        self.model = model
        self.x = x0
        self.P = P0
        self.trace = []

    def __repr__(self):
        """Return string representation of measurement."""
        return "T({}, P)".format(self.x)

    def predict(self, dT):
        """Perform motion prediction."""
        new_x, new_P = self.model(self.x, self.P, dT)
        self.trace.append((new_x, new_P))
        self.x, self.P = new_x, new_P

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

    def bbox(self, nstd=2):
        """Get minimal bounding box approximation."""
        return gaussian_bbox(self.x[0:2], self.P[0:2, 0:2])
