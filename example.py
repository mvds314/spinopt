#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from spinopt import NLOptimizer

dim = 3


def my_easy_func(x, grad):
    if grad.size > 0:
        grad[:] = 2 * (x - np.arange(len(x)))
    x = x - np.arange(len(x))
    return x.dot(x)


# invalid without constraints
opt = NLOptimizer(my_easy_func, np.nan * np.zeros(dim), backend="slsqp")
res = opt.minimize()
assert res.success
assert np.isnan(res.x).all()

# reuse optimizer with other init value
res = opt.minimize(np.zeros(dim))
assert res.success
assert np.allclose(res.x, np.arange(dim), atol=1e-5)

# reuse optimizer and add constraints
A = np.ones((1, dim))
b = np.ones((1, 1))
constraints = [{"type": "eq", "jac": lambda w: A, "fun": lambda w: A.dot(w) - b.squeeze()}]
opt.constraints = constraints
opt.x0 = np.ones(dim)
res = opt.minimize()
assert res.success
assert np.allclose(res.x, [-2 / 3, 1 / 3, 4 / 3], atol=1e-5)
