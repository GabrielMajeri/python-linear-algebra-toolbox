from abc import ABC, abstractmethod
from typing import cast, override

import numpy as np
import numpy.typing as npt
import scipy as sp
import scipy.sparse as sps

type DenseFloatArray = npt.NDArray[np.float64]
# Define a common type alias for dense/sparse arrays
type Array = DenseFloatArray | sps.sparray

# Define a small constant for comparing floating-point numbers
EPSILON = 1e-10


class Preconditioner(ABC):
    """Abstract base class for representing preconditioning matrices."""

    @abstractmethod
    def solve(self, rhs: DenseFloatArray) -> DenseFloatArray:
        """Solves the system `P x = b`, where `P` is the preconditioning matrix
        and `b` is a vector of biases.

        How this is actually implemented depends on the preconditioner type,
        but it's usually very quick.
        """
        pass


class IdentityPreconditioner(Preconditioner):
    "Use the identity matrix as a preconditioner."

    @override
    def solve(self, rhs: DenseFloatArray) -> DenseFloatArray:
        "Solve the system `I_n x = b` by returning the vector `b` directly."
        return rhs


class LUPreconditioner(Preconditioner):
    """Preconditioner based on the LU decomposition."""

    lower: DenseFloatArray
    upper: DenseFloatArray

    def __init__(self, lower: DenseFloatArray, upper: DenseFloatArray):
        self.lower = lower
        self.upper = upper

    @override
    def solve(self, rhs: DenseFloatArray) -> DenseFloatArray:
        # Solve the system `L y = b`
        y = sp.linalg.solve_triangular(self.lower, rhs, lower=True)

        # Solve the system `U x = y`
        x = sp.linalg.solve_triangular(self.upper, y, lower=False)

        return x


class CholeskyPreconditioner(LUPreconditioner):
    """Preconditioner based on the full Cholesky decomposition."""

    lower: DenseFloatArray
    upper: DenseFloatArray

    def __init__(self, lower: DenseFloatArray):
        super().__init__(lower, lower.transpose())


class IncompleteLUPreconditioner(Preconditioner):
    """Preconditioner based on the incomplete LU decomposition."""

    lower: sps.csr_array
    upper: sps.csr_array

    def __init__(self, lower: sps.csr_array, upper: sps.csr_array):
        self.lower = lower
        self.upper = upper

    @override
    def solve(self, rhs: DenseFloatArray) -> DenseFloatArray:
        """Solves a system of linear equations using the (incomplete) lower-upper decomposition."""

        # Solve the system `L y = b`
        y = sps.linalg.spsolve_triangular(self.lower, rhs, lower=True)

        # Solve the system `U x = y`
        x = sps.linalg.spsolve_triangular(self.upper, y, lower=False)

        return cast(DenseFloatArray, x)


class IncompleteCholeskyPreconditioner(IncompleteLUPreconditioner):
    """Preconditioner based on the incomplete Cholesky decomposition."""

    def __init__(self, lower: sps.csr_array) -> None:
        super().__init__(lower, lower.transpose())


def check_preconditioner(
    original_matrix: DenseFloatArray, preconditioner: Preconditioner
) -> bool:
    "Sanity check for a given preconditioner object."

    N = original_matrix.shape[0]
    biases = np.ones(N)

    # Solve the system using the dense preconditioning matrix
    correct_solution = np.linalg.solve(original_matrix, biases)
    preconditioner_solution = preconditioner.solve(biases)

    mse = np.mean(np.square(correct_solution - preconditioner_solution))

    print("Solution as compused using dense matrix inversion:", correct_solution)
    print(
        "Solution as computed by the preconditioner's `solve` method:",
        preconditioner_solution,
    )
    print("Mean squared error:", mse)

    return bool(mse < EPSILON)
