# Fast evaluation of multivariate normal distributions

For optimization, everyone starts out with the [Scipy optimization library](https://docs.scipy.org/doc/scipy/tutorial/optimize.html), but, at some point, you might want to try something else.
[NLOPT](https://nlopt.readthedocs.io/en/latest/) is a great library, but can be quite a hassle rewrite your code to use it.

This package provides a Scipy interface to the NLOPT optimization library. It's aim is not to provide a complete ecosystem which different solvers are available, but merely a way to quickly connect the NLOPT solvers, once you already have something set up with Scipy, or are familiar with how to set something up with Scipy.

## Basic example

```python
import spinopt
```

## Installation

To install from PyPI:

```bash
pip install spinopt
```

To install the latest development version from github:

```bash
pip install git+https://github.com/mvds314/spinopt.git
```

## Development

For development purposes, clone the repo:

```bash
git clone https://github.com/mvds314/spinopt.git
```

Then navigate to the folder containing `setup.py` and run

```bash
pip install -e .
```

to install the package in edit mode.

Run unittests with `pytest`.

## Related software

- [Scipy optimize](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)
- [NLOPT](https://nlopt.readthedocs.io/en/latest/)
