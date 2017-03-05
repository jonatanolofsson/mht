"""Motion and measurement models."""

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

import numpy as np


class ConstantVelocityModel:
    """Constant velocity motion model."""

    def __init__(self, q):
        """Init."""
        self.q = q

    def __call__(self, xprev, Pprev, dT):
        """Step model."""
        x = xprev
        F = np.array([[1, 0, dT, 0],
                       [0, 1, 0, dT],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        Q = np.array([[dT ** 3 / 3, 0,           dT ** 2 / 2, 0],
                      [0,           dT ** 3 / 3, 0,           dT ** 2 / 2],
                      [dT ** 2 / 2, 0,           dT,          0],
                      [0,           dT ** 2 / 2, 0,           dT]]) * self.q
        x = F @ xprev
        P = F @ Pprev @ F.T + Q

        return (x, P)


def position_measurement(x):
    """Velocity measurement model."""
    H = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0]])
    return (H @ x, H)


def velocity_measurement(x):
    """Velocity measurement model."""
    H = np.array([[0, 0, 1, 0],
                   [0, 0, 0, 1]])
    return (H @ x, H)
