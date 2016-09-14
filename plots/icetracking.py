"""Create grd.png plot."""
import os
import sys
import time
import numpy as np
from scipy.cluster.vq import whiten, kmeans, vq
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mht

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
LIB_DIR = '/'.join([GIT_DIR, 'ntnu', 'courses',
                    'remote_sensing', 'python'])
sys.path.insert(0, LIB_DIR)

np.random.seed(1)

from sarproject import \
    image, label, plot_classes, find_point_objects, get_features


def run_kmeans(img):
    """Run kmeans and plot result."""
    features, shape = get_features(img)
    classified = kmeans_classify(features, shape)
    indices, num_objs = label(classified, shape)

    plot_classes(indices, num_objs)
    globpos = find_point_objects(img.lat, img.lon, indices, num_objs)
    return globpos


def kmeans_classify(features, shape, label=True, fill=False):
    """Run the k-means algorithm."""
    print("Starting kmeans")
    whitened = whiten(features)
    init = np.array((whitened.min(0), whitened.mean(0), whitened.max(0)))
    codebook, _ = kmeans(whitened, init)
    classified, _ = vq(whitened, codebook)
    print("Finished kmeans")
    return classified


class TestIcetracking(unittest.TestCase):
    """Test icetracking."""

    def setUp(self):
        """Set up."""
        self.tracker = mht.MHT(k_max=5)

    def test_icetracking(self):
        """Icetracking."""
        pos = run_kmeans(image('grd'))

        scan = [
            mht.Report(
                np.matrix([[plat], [plon]]),
                np.eye(2) * 1e-4,
                mht.models.position_measurement)
            for plat, plon in zip(pos[0], pos[1])
        ]

        for t in range(2):
            t0 = time.time()
            self.tracker.register_scan(
                mht.Scan(mht.sensors.EyeOfMordor(5, 10, 12), scan))
            t1 = time.time()
            print(t, ': Ran tracker in', t1 - t0, 'seconds to generate',
                  len(self.tracker.global_hypotheses), 'hypotheses')


if __name__ == '__main__':
    unittest.main()
