from timeit import timeit

import numpy as np
from linalgtoolbox.random import generate_random_dense_orthogonal_matrix

generator = np.random.default_rng(42)
dimension = 1_000

# Run the function once to ensure it is compiled and cached
_ = generate_random_dense_orthogonal_matrix(generator, dimension)

# Benchmark the function
result = timeit(
    "generate_random_dense_orthogonal_matrix(generator, dimension)",
    globals=globals(),
    number=1,
)

# Report the result
print(f"Took {result:.3f} seconds to generate a large orthonormal matrix")
