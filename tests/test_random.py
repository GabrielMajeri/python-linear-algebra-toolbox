from linalgtoolbox.random import *


SEED = 7

def test_generate_random_dense_matrix():
    generator = np.random.default_rng(SEED)
    matrix = generate_random_dense_matrix(generator, 5)
    assert matrix.shape == (5, 5)
    assert matrix.dtype == np.float64
    assert np.mean(matrix) < 1e-10
    assert np.mean(matrix ** 2) - 1 < 1e-10
