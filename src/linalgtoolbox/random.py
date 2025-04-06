"Helpers for generating random matrices of various kinds."

from typing import Literal, cast

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


def generate_random_dense_symmetric_matrix(
    generator: np.random.Generator, dimension: int
) -> npt.NDArray[np.float64]:
    """Generates a dense symmetric matrix of the given size
    with random (normally distributed) contents.
    """
    random_matrix = generate_random_dense_matrix(generator, dimension)

    # Cost: O(n^2)
    random_symmetric_matrix = 0.5 * (random_matrix + random_matrix.T)

    return random_symmetric_matrix


def generate_random_dense_spd_matrix(
    generator: np.random.Generator, dimension: int
) -> npt.NDArray[np.float64]:
    """Generates a dense symmetric and positive-definite matrix of the given size
    with random (normally distributed) contents.
    """
    random_symmetric_matrix = generate_random_dense_symmetric_matrix(
        generator, dimension
    )

    return random_symmetric_matrix + dimension * np.eye(dimension)


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


def generate_random_sparse_spd_matrix(
    generator: np.random.Generator,
    dimension: int,
    density: float = 0.01,
    format: SPFormat = "csc",
) -> sps.sparray:
    """Generates a random symmetric positive-definite matrix,
    with random (normally distributed) contents (except on the main diagonal).
    """
    random_symmetric_matrix = generate_random_sparse_symmetric_matrix(
        generator, dimension, density=density, format=format
    )

    random_symmetric_matrix = cast(sps.csr_array, random_symmetric_matrix)

    identity_matrix = sps.eye_array(dimension, dimension, format=format)
    identity_matrix = cast(sps.csr_array, identity_matrix)

    # Add a large diagonal matrix (n * I_n) to the random symmetric matrix,
    # to ensure it becomes diagonally dominant. This will ensure it's also positive definite.
    random_spd_matrix = random_symmetric_matrix + dimension * identity_matrix

    return cast(sps.sparray, random_spd_matrix)


def generate_random_vector(
    generator: np.random.Generator, rows: int
) -> npt.NDArray[np.float64]:
    """Generates a random vector with random (normally distributed) contents."""
    random_vector = generator.normal(loc=0.0, scale=1.0, size=rows)

    return random_vector
