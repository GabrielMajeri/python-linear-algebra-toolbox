from collections.abc import Callable

import numpy as np
import numpy.typing as npt

type Vector = npt.NDArray[np.float64]

type InnerProduct = Callable[[Vector, Vector], float]


def dot_product(a: Vector, b: Vector) -> float:
    """Returns the usual (Euclidean) scalar product
    between the vectors `a` and `b`.
    """
    return (a.T @ b).item()


def construct_matrix_induced_inner_product(
    matrix: npt.NDArray[np.float64],
) -> InnerProduct:
    """Constructs the inner product induced by the matrix given as parameter,
    that is `(a, b)_{M} = a^{\\intercal} M b`.
    """
    assert np.all(np.equal(matrix.T, matrix)), "Matrix must be symmetric"
    if __debug__:
        assert np.all(np.linalg.eigvals(matrix) > 0), "Matrix must be positive definite"

    def inner_product(a: Vector, b: Vector) -> float:
        return (a.T @ matrix @ b).item()

    return inner_product
