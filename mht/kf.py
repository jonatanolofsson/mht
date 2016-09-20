"""Kalman Filter implementation for MHT target."""

from copy import deepcopy
from math import log as ln
from math import sqrt, pi
from numpy.linalg import det
import numpy as np
from numpy.linalg import inv

from . import models
from .utils import gaussian_bbox


class DefaultTargetInit:
    """Default target initiator."""

    def __init__(self, q):
        """Init."""
        self.q = q

    def __call__(self, report, parent=None):
        """Init new target from report."""
        if parent is None:
            model = models.ConstantVelocityModel(self.q)
            x0 = np.array([report.z[0], report.z[1], 0.0, 0.0])
            P0 = np.eye(4) * 2
            return KFilter(model, x0, P0)
        elif parent.is_new():
            model = models.ConstantVelocityModel(self.q)
            x0 = np.array([report.z[0],
                           report.z[1],
                           report.z[0] - parent.filter.x[0],
                           report.z[1] - parent.filter.x[1]])
            P0 = np.eye(4)
            return KFilter(model, x0, P0)
        else:
            return deepcopy(parent.filter)


class KFilter:
    """Kalman-filter target."""

    def __init__(self, model, x0, P0):
        """Init."""
        self.model = model
        self.x = x0
        self.P = P0
        self.trace = [(x0, P0)]

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
        S = H @ self.P @ H.T + m.R
        SI = inv(S)
        K = self.P @ H.T @ SI
        self.x += K @ dz
        self.P -= K @ H @ self.P

        score = dz.T @ SI @ dz / 2.0 + ln(2 * pi * sqrt(det(S)))
        return float(score)

    def nll(self, m):
        """Get the nll score of assigning a measurement to the filter."""
        zhat, H = m.mfn(self.x)
        dz = m.z - zhat
        S = H @ self.P @ H.T + m.R
        score = dz.T @ inv(S) @ dz / 2.0 + ln(2 * pi * sqrt(det(S)))
        return float(score)

    def bbox(self, nstd=2):
        """Get minimal bounding box approximation."""
        return gaussian_bbox(self.x[0:2], self.P[0:2, 0:2])
