import numpy as np

from marcd.utils import ledoit_wolf_shrinkage


def test_ledoit_wolf_shrinkage_shape():
    cov = np.eye(5)
    shrunk = ledoit_wolf_shrinkage(cov, delta=0.1)
    assert shrunk.shape == cov.shape
