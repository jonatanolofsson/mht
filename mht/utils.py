"""Util methods."""

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


def eigsorted(cov):
    """Return eigenvalues, sorted."""
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def cov_ellipse(cov, nstd):
    """Get the covariance ellipse."""
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    return width, height, theta


def gaussian_bbox(x, P, nstd=2):
    """Return boudningbox for gaussian."""
    width, height, theta = cov_ellipse(P, nstd)
    width, height = width / 2, height / 2
    ux = width * cos(theta)
    uy = width * sin(theta)
    vx = height * cos(theta + pi/2)
    vy = height * sin(theta + pi/2)

    dx = sqrt(ux*ux + vx*vx)
    dy = sqrt(uy*uy + vy*vy)

    return (x[0] - dx, x[0] + dx, x[1] - dy, x[1] + dy)
