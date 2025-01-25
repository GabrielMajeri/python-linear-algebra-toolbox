"Helpers for generating random matrices of various kinds."

from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sps


SPFormat = Literal["csc", "csr", "coo", "bsr"]


def generate_random_dense_matrix(
    generator: np.random.Generator, dimension: int
) -> npt.NDArray[np.float64]:
    """Generates a dense matrix of the given size
    with random (normally distributed) contents.
    """
    shape = (dimension, dimension)
    return generator.normal(size=shape)


def generate_random_dense_orthogonal_matrix(
    generator: np.random.Generator, dimension: int
) -> npt.NDArray[np.float64]:
    """Generates a random orthogonal matrix (dense) of the given size."""
    # Generate an arbitrary random matrix
    random_matrix = generate_random_dense_matrix(generator, dimension)
    # Use the QR decomposition to get an orthogonal matrix
    Q, _ = np.linalg.qr(random_matrix, mode="complete")
    return Q


def generate_random_sparse_symmetric_matrix(
    generator: np.random.Generator,
    dimension: int,
    density: float = 0.01,
    format: SPFormat = "csc",
) -> sps.sparray:
    """Generates a symmetric matrix of the given size
    with random (normally distributed) contents.
    """
    shape = (dimension, dimension)
    random_sparse_matrix = sps.random_array(
        shape, density=density, format=format, dtype=np.float64, rng=generator
    )

    # Cost: O(n^2)
    random_symmetric_matrix = 0.5 * (random_sparse_matrix + random_sparse_matrix.T)

    return random_symmetric_matrix
