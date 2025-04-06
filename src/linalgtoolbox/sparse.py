from typing import cast, final

import numpy as np
import scipy.sparse as sps


@final
class SparseAproximateInverse:
    """Helper class for computing the approximate inverse of a sparse matrix.

    ## References

    The method is described and analyzed formally [in this paper](https://dl.acm.org/doi/10.1137/S1064827594270415).

    The implementation is based on [this code](https://tbetcke.github.io/hpc_lecture_notes/it_solvers4.html).
    """

    _matrix: sps.sparray
    _identity: sps.sparray
    _alpha: np.floating
    _approximate_inverse: sps.sparray

    def __init__(self, matrix: sps.sparray) -> None:
        # Cast the input matrix to a concrete array layout
        # in order to alleviate typing issues.
        matrix = cast(sps.csc_array, matrix)
        self._matrix = matrix
        self._n = matrix.shape[0]
        self._identity = sps.eye_array(self._n)

        self._alpha = 2 / sps.linalg.norm(matrix @ matrix.T, ord=1)
        self._approximate_inverse = self._alpha * matrix

    @property
    def approximate_inverse(self) -> sps.sparray:
        return self._approximate_inverse

    def step(self) -> None:
        matrix = cast(sps.csc_array, self._matrix)
        identity = cast(sps.csc_array, self._identity)
        approximate_inverse = cast(sps.csc_array, self._approximate_inverse)

        C = matrix @ approximate_inverse
        G = identity - C
        AG = matrix @ G

        trace = cast(float, (G.T @ AG).diagonal().sum())
        self._alpha = trace / np.linalg.norm(AG.data) ** 2
        self._approximate_inverse = approximate_inverse + self._alpha * G
