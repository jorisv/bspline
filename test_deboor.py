from typing import List

import numpy as np
from bspline import Bspline


def deBoor(k: int, x: int, t, c, p: int):
    d = [c[j + k - p] for j in range(0, p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            alpha = (x - t[j + k - p]) / (t[j + 1 + k - r] - t[j + k - p])
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


def deBoorBasisSparse(
    start_knot: int, t: float, knots: List[float], p: int, ignore: int
):
    d = [[0.0] * (p + 1 - j) for j in range(0, p + 1)]
    d[0] = [1.0] * (p + 1)
    for i in range(ignore):
        d[0][i] = 0.0
    for r in range(p):
        for j in range(p - r):
            _r = r + 1
            _i = start_knot - p + _r + j
            alpha = (t - knots[_i]) / (knots[_i + 1 + p - _r] - knots[_i])
            d[r + 1][j] = (1.0 - alpha) * d[r][j] + alpha * d[r][j + 1]
    return d[-1]


def deBoorBasisSparseV2(
    start_knot: int, t: float, knots: List[float], p: int, ignore: int
):
    d = [[0.0] * (p + 1 - j) for j in range(0, p + 1)]
    d[0] = [1.0] * (p + 1)
    for i in range(ignore):
        d[0][i] = 0.0
    for r in range(ignore):
        _r = r + 1
        _i = start_knot - p + _r + ignore - _r
        alpha = (t - knots[_i]) / (knots[_i + 1 + p - _r] - knots[_i])
        d[r + 1][ignore - _r] = alpha * d[r][ignore - _r + 1]
    for r in range(p):
        _r = r + 1
        # TODO better ?
        start = max(0, ignore - _r + 1)
        for j in range(start, p - r):
            _i = start_knot - p + _r + j
            alpha = (t - knots[_i]) / (knots[_i + 1 + p - _r] - knots[_i])
            d[r + 1][j] = (1.0 - alpha) * d[r][j] + alpha * d[r][j + 1]
    return d[-1]


knot_vector = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
degree = 4
order = degree + 1
nb_control = len(knot_vector) - order
basis = Bspline(knot_vector, degree)

x, N, N_cum = basis.compute(False)

print(nb_control)
print(x.shape, N.shape, N_cum.shape)

start_knot = 4
index = 250
print(f"{x[index]=}")
print(N[:, index])
print(N_cum[:, index])

print()
print(
    "1, 1, 1, 1, 1", deBoor(start_knot, x[index], knot_vector, [1, 1, 1, 1, 1], degree)
)
print(
    "0, 1, 1, 1, 1", deBoor(start_knot, x[index], knot_vector, [0, 1, 1, 1, 1], degree)
)
print(
    "0, 0, 1, 1, 1", deBoor(start_knot, x[index], knot_vector, [0, 0, 1, 1, 1], degree)
)
print(
    "0, 0, 0, 1, 1", deBoor(start_knot, x[index], knot_vector, [0, 0, 0, 1, 1], degree)
)
print(
    "0, 0, 0, 0, 1", deBoor(start_knot, x[index], knot_vector, [0, 0, 0, 0, 1], degree)
)
print()
print("1, 1, 1, 1, 1", deBoorBasisSparse(start_knot, x[index], knot_vector, degree, 0))
print("0, 1, 1, 1, 1", deBoorBasisSparse(start_knot, x[index], knot_vector, degree, 1))
print("0, 0, 1, 1, 1", deBoorBasisSparse(start_knot, x[index], knot_vector, degree, 2))
print("0, 0, 0, 1, 1", deBoorBasisSparse(start_knot, x[index], knot_vector, degree, 3))
print("0, 0, 0, 0, 1", deBoorBasisSparse(start_knot, x[index], knot_vector, degree, 4))
print()
print(
    "1, 1, 1, 1, 1", deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 0)
)
print(
    "0, 1, 1, 1, 1", deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 1)
)
print(
    "0, 0, 1, 1, 1", deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 2)
)
print(
    "0, 0, 0, 1, 1", deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 3)
)
print(
    "0, 0, 0, 0, 1", deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 4)
)
