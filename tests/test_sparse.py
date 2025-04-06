from typing import cast

import numpy as np
import scipy.sparse as sps

from linalgtoolbox.random import generate_random_sparse_spd_matrix
from linalgtoolbox.sparse import SparseAproximateInverse


def test_sparse_approximate_inverse_works_for_small_matrix():
    matrix = np.array([[1.0, 2.0, 13.0], [-1.0, 4.0, 4.0], [5.0, -3.0, 7.0]])
    n = matrix.shape[0]

    assert np.abs(cast(float, np.linalg.det(matrix))) > 1

    matrix = sps.csc_array(matrix)

    spai = SparseAproximateInverse(matrix)

    spai.step()

    approximate_inverse = cast(sps.csc_array, spai.approximate_inverse)
    identity = sps.eye_array(n)

    assert sps.linalg.norm(approximate_inverse * matrix - identity) < 5


def test_sparse_approximate_inverse_converges_for_large_matrix():
    generator = np.random.default_rng(7)
    n = 100
    matrix = generate_random_sparse_spd_matrix(generator, n)
    matrix = cast(sps.csc_array, matrix)

    spai = SparseAproximateInverse(matrix)

    for _ in range(5):
        spai.step()

    approximate_inverse = cast(sps.csc_array, spai.approximate_inverse)
    identity = sps.eye_array(n)

    assert sps.linalg.norm(approximate_inverse * matrix - identity) < 5
