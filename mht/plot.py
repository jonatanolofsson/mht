"""Helper functions for MHT plots."""

import matplotlib.colors
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

CMAP = matplotlib.colors.ListedColormap(RandomState(0).rand(256, 3))


def plot_trace(trace, c=0, covellipse=True, **kwargs):
    """Plot single trace."""
    x = []
    y = []
    for track in trace:
        pos = (float(track.filter.x[0]), float(track.filter.x[1]))
        x.append(pos[0])
        y.append(pos[1])
        if covellipse:
            ca = plot_cov_ellipse(track.filter.P[0:2, 0:2], pos)
            ca.set_alpha(0.3)
            ca.set_facecolor(CMAP(c))
    plt.plot(x, y, marker='*', color=CMAP(c))


def plot_hyptrace(gh, cseed=0, covellipse=True, **kwargs):
    """Plot hypothesis trace."""
    for tr in gh.tracks:
        plot_trace(tr.trace(), tr.target._id + cseed, covellipse, **kwargs)


def plot_cov_ellipse(cov, pos, nstd=2, **kwargs):
    """Plot confidence ellipse."""
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    plt.gca().add_artist(ellip)
    return ellip


def plot_hypothesis(gh, cseed=0, covellipse=True):
    """Plot targets."""
    for tr in gh.tracks:
        pos = (tr.filter.x[0], tr.filter.x[1])
        plt.scatter(*pos, c=tr.target._id+cseed, cmap=CMAP, edgecolors='k')
        if covellipse:
            ca = plot_cov_ellipse(tr.filter.P[0:2, 0:2], pos)
            ca.set_alpha(0.5)
            ca.set_facecolor(CMAP(tr.target._id + cseed))


def plot_scan(scan):
    """Plot reports from scan."""
    plt.plot([float(r.z[0]) for r in scan.reports],
             [float(r.z[1]) for r in scan.reports],
             marker='+', color='r', linestyle='None')
