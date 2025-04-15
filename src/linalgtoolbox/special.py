import numpy as np
import numpy.typing as npt

type Matrix = npt.NDArray[np.float64]


def construct_hilbert_matrix(n: int, m: int | None = None) -> Matrix:
    """Constructs the [Hilbert matrix](https://en.wikipedia.org/wiki/Hilbert_matrix)
    of a given dimension.

    If `m` is not given, it is assumed to be equal to `n`
    and the n-by-n Hilbert matrix will be constructed and returned.
    """
    if m is None:
        m = n

    assert n > 0 and m > 0

    H = np.empty(shape=(n, m), dtype=np.float64)

    for i in range(n):
        for j in range(m):
            H[i, j] = 1 / (1 + i + j)

    return H
