"""Helper functions for MHT plots."""

import matplotlib.colors
from numpy.random import RandomState
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

from .utils import cov_ellipse

CMAP = matplotlib.colors.ListedColormap(RandomState(0).rand(256, 3))


def plot_trace(trace, c=0, covellipse=True, **kwargs):
    """Plot single trace."""
    xs = []
    ys = []
    for x, P in trace:
        pos = (float(x[0]), float(x[1]))
        xs.append(pos[0])
        ys.append(pos[1])
        if covellipse:
            ca = plot_cov_ellipse(P[0:2, 0:2], pos)
            ca.set_alpha(0.3)
            ca.set_facecolor(CMAP(c))
    plt.plot(xs, ys, marker='*', color=CMAP(c))


def plot_hyptrace(gh, cseed=0, covellipse=True, **kwargs):
    """Plot hypothesis trace."""
    for tr in gh.tracks:
        plot_trace(tr.filter.trace, tr.target._id + cseed, covellipse,
                   **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """Plot confidence ellipse."""
    r1, r2, theta = cov_ellipse(cov, nstd)
    ellip = Ellipse(xy=pos, width=2*r1, height=2*r2, angle=theta, **kwargs)

    plt.gca().add_artist(ellip)
    return ellip


def plot_hypothesis(gh, cseed=0, covellipse=True, **kwargs):
    """Plot targets."""
    options = {'edgecolors': 'k'}
    options.update(kwargs)

    for tr in gh.tracks:
        pos = (tr.filter.x[0], tr.filter.x[1])
        plt.scatter(*pos, c=tr.target._id+cseed, cmap=CMAP, **options)
        if covellipse:
            ca = plot_cov_ellipse(tr.filter.P[0:2, 0:2], pos)
            ca.set_alpha(0.5)
            ca.set_facecolor(CMAP(tr._id + cseed))


def plot_scan(scan, covellipse=True, **kwargs):
    """Plot reports from scan."""
    options = {
        'marker': '+',
        'color': 'r',
        'linestyle': 'None'
    }
    options.update(kwargs)
    plt.plot([float(r.z[0]) for r in scan.reports],
             [float(r.z[1]) for r in scan.reports], **options)
    if covellipse:
        for r in scan.reports:
            ca = plot_cov_ellipse(r.R[0:2, 0:2], r.z[0:2])
            ca.set_alpha(0.1)
            ca.set_facecolor(options['color'])


def plot_bbox(obj, cseed=0, **kwargs):
    """Plot bounding box."""
    id_ = getattr(obj, '_id', 0)
    options = {
        'alpha': 0.3,
        'color': CMAP(id_ + cseed)
    }
    options.update(kwargs)
    bbox = obj.bbox()
    plt.gca().add_patch(Rectangle(
        (bbox[0], bbox[2]), bbox[1] - bbox[0], bbox[3] - bbox[2],
        **options))
