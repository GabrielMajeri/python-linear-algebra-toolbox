from typing import cast

import numpy as np
import scipy.sparse as sps

from linalgtoolbox.random import (
    generate_random_dense_matrix,
    generate_random_dense_orthogonal_matrix,
    generate_random_sparse_symmetric_matrix,
    generate_random_sparse_spd_matrix,
)


EPSILON = 1e-10
SEED = 7


def test_generate_random_dense_matrix() -> None:
    generator = np.random.default_rng(SEED)
    matrix = generate_random_dense_matrix(generator, 5)
    assert matrix.shape == (5, 5)
    assert matrix.dtype == np.float64
    assert np.mean(matrix) < EPSILON
    assert np.mean(matrix**2) - 1 < EPSILON


def test_generate_random_sparse_symmetric_matrix() -> None:
    generator = np.random.default_rng(SEED)
    dimension = 7
    matrix = generate_random_sparse_symmetric_matrix(generator, dimension)
    matrix = cast(sps.csc_array, matrix)
    assert matrix.shape == (dimension, dimension)
    assert matrix.dtype == np.float64
    assert sps.linalg.norm(matrix - matrix.T) < EPSILON


def test_generate_random_sparse_spd_matrix() -> None:
    generator = np.random.default_rng(SEED)
    dimension = 15
    matrix = generate_random_sparse_spd_matrix(generator, dimension)
    matrix = cast(sps.csc_array, matrix)
    assert matrix.shape == (dimension, dimension)
    assert matrix.dtype == np.float64
    assert sps.linalg.norm(matrix - matrix.T) < EPSILON
    assert np.all(sps.linalg.eigs(matrix, k=dimension - 2)[0] > 0)


def test_generate_random_dense_orthogonal_matrix() -> None:
    generator = np.random.default_rng(SEED)
    dimension = 10
    orthogonal_matrix = generate_random_dense_orthogonal_matrix(generator, dimension)
    assert orthogonal_matrix.shape == (dimension, dimension)
    assert (
        np.linalg.norm(orthogonal_matrix @ orthogonal_matrix.T - np.eye(dimension))
        < EPSILON
    )
