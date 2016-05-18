"""Helper functions for MHT plots."""

import matplotlib.colors
from numpy.random import RandomState
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

CMAP = matplotlib.colors.ListedColormap(RandomState(0).rand(256, 3))


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


def plot_hypothesis(gh):
    """Plot targets."""
    for c, track in enumerate(gh.tracks):
        pos = (track.filter.x[0], track.filter.x[1])
        ca = plot_cov_ellipse(track.filter.P[0:2, 0:2], pos)
        ca.set_alpha(0.5)
        ca.set_facecolor(CMAP(c))
        plt.scatter(*pos, c=c, cmap=CMAP, edgecolors='k')
    plt.plot([float(u.z[0]) for u in gh.unassigned],
             [float(u.z[1]) for u in gh.unassigned],
             marker='*', color='r', linestyle='None')
