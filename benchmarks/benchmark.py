from timeit import timeit

import numpy as np
from linalgtoolbox.random import *

generator = np.random.default_rng(42)
dimension = 1_000

# Benchmark the function
result = timeit('generate_random_dense_orthogonal_matrix(generator, dimension)', globals=globals(), number=1)

# Report the result
print(f'Took {result:.3f} seconds to generate a large orthonormal matrix')
