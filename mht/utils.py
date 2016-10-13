"""Util methods."""

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

LARGE = 10000
import numpy as np
from math import sin, cos, pi, sqrt


class PrioItem:
    """Item storable in PriorityQueue."""

    def __init__(self, prio, data):
        """Init."""
        self.prio = prio
        self.data = data

    def __lt__(self, b):
        """lt comparison."""
        return self.prio < b.prio


def anyitem(iterable):
    """Retrieve 'first' item from set."""
    try:
        return next(iter(iterable))
    except StopIteration:
        return None


def connected_components(connections):
    """Get all connected components."""
    seen = set()

    def component(node):
        nodes = {node}
        while nodes:
            node = nodes.pop()
            seen.add(node)
            nodes |= connections[node] - seen
            yield node
    for node in list(connections.keys()):
        if node not in seen:
            yield set(component(node))


def overlap(a, b):
    """Check if boundingboxes overlap."""
    return (a[1] >= b[0] and a[0] <= b[1] and
            a[3] >= b[2] and a[2] <= b[3])


def overlap_pa(a, b):
    """Return percentage of bbox a being in b."""
    intersection = max(0, min(a[1], b[1]) - max(a[0], b[0])) \
        * max(0, min(a[3], b[3]) - max(a[2], b[2]))
    aa = (a[1] - a[0]) * (a[3] - a[2])
    return intersection / aa


def eigsorted(cov):
    """Return eigenvalues, sorted."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def cov_ellipse(cov, nstd):
    """Get the covariance ellipse."""
    vals, vecs = eigsorted(cov)
    r1, r2 = nstd * np.sqrt(vals)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    return r1, r2, theta


def gaussian_bbox(x, P, nstd=2):
    """Return boudningbox for gaussian."""
    r1, r2, theta = cov_ellipse(P, nstd)
    ux = r1 * cos(theta)
    uy = r1 * sin(theta)
    vx = r2 * cos(theta + pi/2)
    vy = r2 * sin(theta + pi/2)

    dx = sqrt(ux*ux + vx*vx)
    dy = sqrt(uy*uy + vy*vy)

    return (float(x[0] - dx),
            float(x[0] + dx),
            float(x[1] - dy),
            float(x[1] + dy))


def within(p, bbox):
    """Check if point is within bbox."""
    return ((bbox[0] <= p[0] <= bbox[1]) and (bbox[2] <= p[1] <= bbox[3]))
