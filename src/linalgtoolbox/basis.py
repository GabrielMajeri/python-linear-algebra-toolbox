from math import sqrt

import numpy as np

from linalgtoolbox.product import Vector, InnerProduct


def orthonormalize(
    vectors: list[Vector], inner_product: InnerProduct = np.dot
) -> list[Vector]:
    n = len(vectors)

    def norm(v: Vector) -> np.float64:
        return np.float64(sqrt(inner_product(v, v)))

    def project(target: Vector, v: Vector) -> Vector:
        return (inner_product(target, v) / inner_product(target, target)) * target

    intermediate: list[Vector] = []
    basis: list[Vector] = []

    for i in range(n):
        v = vectors[i]
        assert len(v) == n, f"Expected {n} vectors of common dimension {n}"

        u = v
        for j in range(i):
            w = intermediate[j]
            u -= project(w, v)

        intermediate.append(u)

        e = u / norm(u)
        basis.append(e)

    return basis
