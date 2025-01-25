# Linear algebra toolbox for Python

## Summary

This repository contains a set of classes and functions for doing [numerical linear algebra](https://en.wikipedia.org/wiki/Numerical_linear_algebra) in [Python](https://www.python.org/). It is meant to bridge the gap from more fundamental libraries such as [`numpy.linalg`](https://numpy.org/doc/stable/reference/routines.linalg.html), [`scipy.linalg`](https://docs.scipy.org/doc/scipy/reference/linalg.html) and [`scipy.sparse.linalg`](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html) (which we depend on and extend) and more advanced algorithms used in research.

The guiding principles for developing this library are:

- The code should be **performant**, aiming to be as fast as if you'd written it yourself in a lower-level language.

- The code should be **well-documented**, **concise** and easily **readable**.

- Existing high-quality code and libraries should be **reused** as much as possible, avoiding the ["not invented here"](https://en.wikipedia.org/wiki/Not_invented_here) syndrome.

- The codebase should be **thoroughly tested** to ensure it is reliable in a variety of situations.

In order to achieve the first goal, we rely on efficient implementations of fundamental operations and data structres in [NumPy](https://numpy.org/) and [SciPy](https://scipy.org/), as well as on [Numba](https://numba.pydata.org/) to speed up functions by JIT-compiling them to native code. For the second goal, we try to decorate our functions with type information and use [Ruff](https://github.com/astral-sh/ruff) to perform linting and static type checking. For the fourth goal, we maintain a large test suite, using the [pytest](https://docs.pytest.org/en/stable/) framework.

## Development instructions

This section describes the steps you need to take to be able to work on and extend this package.
It is not necessary to follow them if you only plan to use the library in your own research.

### Setting up a local development environment

This library requires a relatively modern version of [Python](https://www.python.org/) (3.12 or newer). One of the easiest ways to install and manage your Python distribution is by using a tool like [Anaconda](https://www.anaconda.com) (in particular, [Miniconda](https://docs.anaconda.com/miniconda/) is often more than enough). Alternatively, you could use Python's built-in [virtual environment support](https://docs.python.org/3/library/venv.html) (the `venv` module).

Once you've got a basic Python installation set up, you can install this package in editable/development mode by using [`pip`](https://pypi.org/project/pip/):

```sh
pip install --editable .
```

### Linting the code

The recommended tool to efficiently format, lint and type-check the code is [Ruff](https://github.com/astral-sh/ruff?tab=readme-ov-file).

Once installed, you can do

```sh
ruff check
```

If you're using [VS Code](https://code.visualstudio.com/) for development, you can also install [the official Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff), for a better integration.

### Running the benchmarks

To run the benchmarking code, use

```sh
python3 benchmarks/benchmark.py
```

### Running all of the tests

To run all of the library's tests, install [`pytest`](https://docs.pytest.org/en/stable/) and then run

```sh
pytest tests/
```

To obtain test coverage reports, install the [`pytest-cov`](https://pypi.org/project/pytest-cov/) plugin and then run:

```sh
pytest --cov=linalgtoolbox tests/
```

## License

The source code in this repository is available under the permissive MIT license. See the [`LICENSE.txt`](LICENSE.txt) file for more information.

## Acknowledgements

A lot of these functions where collected from various places on the internet
(with credits mentioned where they are due) or developed from scratch
with the help and supervision of prof. [Cristian Rusu](https://cs.unibuc.ro/~crusu/).
