"Helpers for generating random matrices of various kinds."

import numpy as np
import scipy as sp


def generate_random_dense_matrix(generator: np.random.Generator, dimension: int) -> np.ndarray:
    """Generates a dense matrix of the given size
    with random (normally distributed) contents.
    """
    shape = (dimension, dimension)
    return generator.normal(size=shape)

def generate_random_dense_orthogonal_matrix(
        generator: np.random.Generator, dimension: int) -> np.ndarray:
    """Generates a random orthogonal matrix (dense) of the given size."""
    # Generate an arbitrary random matrix
    random_matrix = generate_random_dense_matrix(generator, dimension)
    # Use the QR decomposition to get an orthogonal matrix
    Q, _ = np.linalg.qr(random_matrix, mode='complete')
    return Q

def generate_random_sparse_symmetric_matrix(
        generator: np.random.Generator, dimension: int,
        density: float = 0.01, format = 'csc') -> sp.sparse.sparray:
    """Generates a symmetric matrix of the given size
    with random (normally distributed) contents.
    """
    shape = (dimension, dimension)
    random_sparse_matrix = sp.sparse.random_array(
        shape, density=density, format=format,
        dtype=np.float32, random_state=generator
    )

    # Cost: O(n^2)
    random_symmetric_matrix = 0.5 * (random_sparse_matrix + random_sparse_matrix.T)

    return random_symmetric_matrix
