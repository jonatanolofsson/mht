"""Kalman Filter implementation for MHT target."""

"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from copy import deepcopy
from math import log as ln
from math import sqrt, pi
from numpy.linalg import det
import numpy as np
from numpy.linalg import inv
from scipy.linalg import block_diag
import numbers

from . import models
from .utils import gaussian_bbox


class DefaultTargetInit:
    """Default target initiator."""

    def __init__(self, q, pv, dT=1):
        """Init."""
        self.q = q
        self.pv = np.eye(2) * pv if isinstance(pv, numbers.Number) else pv
        self.dT = dT

    def __call__(self, report, parent=None):
        """Init new target from report."""
        if parent is None:
            model = models.ConstantVelocityModel(self.q)
            x0 = np.array([report.z[0], report.z[1], 0.0, 0.0])
            P0 = block_diag(report.R, self.pv)
            return KFilter(model, x0, P0)
        # elif parent.is_new():
            # model = models.ConstantVelocityModel(self.q)
            # x0 = np.array([report.z[0],
                           # report.z[1],
                           # (report.z[0] - parent.filter.x[0])/self.dT,
                           # (report.z[1] - parent.filter.x[1])/self.dT])
            # P0 = block_diag(report.R, self.pv)
            # return KFilter(model, x0, P0)
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
        self._calc_bbox()

    def __repr__(self):
        """Return string representation of measurement."""
        return "T({}, P)".format(self.x)

    def predict(self, dT):
        """Perform motion prediction."""
        new_x, new_P = self.model(self.x, self.P, dT)
        self.trace.append((new_x, new_P))
        self.x, self.P = new_x, new_P

        self._calc_bbox()

    def correct(self, r):
        """Perform correction (measurement) update."""
        zhat, H = r.mfn(self.x)
        dz = r.z - zhat
        S = H @ self.P @ H.T + r.R
        SI = inv(S)
        K = self.P @ H.T @ SI
        self.x += K @ dz
        self.P -= K @ H @ self.P

        score = dz.T @ SI @ dz / 2.0 + ln(2 * pi * sqrt(det(S)))

        self._calc_bbox()

        return float(score)

    def nll(self, r):
        """Get the nll score of assigning a measurement to the filter."""
        zhat, H = r.mfn(self.x)
        dz = r.z - zhat
        S = H @ self.P @ H.T + r.R
        score = dz.T @ inv(S) @ dz / 2.0 + ln(2 * pi * sqrt(det(S)))
        return float(score)

    def _calc_bbox(self, nstd=2):
        """Calculate minimal bounding box approximation."""
        self._bbox = gaussian_bbox(self.x[0:2], self.P[0:2, 0:2])

    def bbox(self):
        """Get minimal bounding box approximation."""
        return self._bbox
