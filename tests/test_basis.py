import numpy as np

from linalgtoolbox.basis import orthonormalize
from linalgtoolbox.product import (
    InnerProduct,
    construct_matrix_induced_inner_product,
)


def test_orthonormalize_small_set_of_vectors():
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.0]),
        np.array([2.0, 0.0, 3.0]),
    ]

    basis = orthonormalize(vectors)

    assert np.all(np.equal(basis[0], vectors[0]))
    assert np.all(np.equal(basis[1], np.array([0.0, 1.0, 0.0])))
    assert np.all(np.equal(basis[2], np.array([0.0, 0.0, 1.0])))


def test_orthonormalize_with_respect_to_spd_matrix_induced_product():
    vectors = [
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 2.0, 0.0]),
        np.array([2.0, 0.0, 3.0]),
    ]

    A = np.array(
        [
            [9.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    matrix_inner_product: InnerProduct = construct_matrix_induced_inner_product(A)
    basis = orthonormalize(vectors, matrix_inner_product)

    assert np.all(np.equal(basis[0], np.array([1 / 3, 0.0, 0.0])))
    assert np.all(np.equal(basis[1], np.array([0.0, 1 / 2, 0.0])))
    assert np.all(np.equal(basis[2], np.array([0.0, 0.0, 1.0])))
