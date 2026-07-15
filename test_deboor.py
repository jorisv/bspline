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
    # Iterate over all degree
    for r in range(p):
        current_degree = p - (r + 1)
        assert current_degree >= 0
        # Iterate over all non zero knot vector for a specific degree
        for j in range(current_degree + 1):
            current_knot = start_knot - current_degree + j
            assert current_knot >= 0
            last_knot = current_knot + current_degree + 1
            assert last_knot >= 0
            alpha = (t - knots[current_knot]) / (knots[last_knot] - knots[current_knot])
            d[r + 1][j] = (1.0 - alpha) * d[r][j] + alpha * d[r][j + 1]
    return d[-1]


def deBoorBasisSparseV2(
    start_knot: int, t: float, knots: List[float], p: int, ignore: int
):
    d = [[0.0] * (p + 1 - j) for j in range(0, p + 1)]
    d[0] = [1.0] * (p + 1)
    if ignore > 0:
        d[0][ignore - 1] = 0.0
    # Evaluate basis function with only non zero right coefficient
    first_pass_start_knot = start_knot - p + ignore
    assert first_pass_start_knot >= 0
    for r in range(ignore):
        current_degree = p - (r + 1)
        assert current_degree >= 0
        last_knot = first_pass_start_knot + current_degree + 1
        assert last_knot >= 0
        alpha = (t - knots[first_pass_start_knot]) / (
            knots[last_knot] - knots[first_pass_start_knot]
        )
        d[r + 1][ignore - (r + 1)] = alpha * d[r][ignore - r]
    # Iterate over all degree
    for r in range(p):
        current_degree = p - (r + 1)
        start = max(0, ignore - r)
        # Iterate over all non zero knot vector for a specific degree
        for j in range(start, current_degree + 1):
            current_knot = start_knot - current_degree + j
            assert current_knot >= 0
            last_knot = current_knot + current_degree + 1
            assert last_knot >= 0
            alpha = (t - knots[current_knot]) / (knots[last_knot] - knots[current_knot])
            d[r + 1][j] = (1.0 - alpha) * d[r][j] + alpha * d[r][j + 1]
    return d[-1]


def deBoorBasis(root_basis: int, t: float, knots: List[float], p: int):
    d = np.zeros((p + 1, p + 1))

    d[0, 0] = 1.0
    # Evaluate diagonal part
    for previous_degree in range(p):
        current_degree = previous_degree + 1
        left_basis = root_basis - current_degree
        left_basis_start_knot = left_basis + 1
        left_basis_end_knot = left_basis_start_knot + current_degree
        left_basis_alpha = (knots[left_basis_end_knot] - t) / (
            knots[left_basis_end_knot] - knots[left_basis_start_knot]
        )
        d[current_degree, 0] = left_basis_alpha * d[previous_degree, 0]

        right_basis = root_basis
        right_basis_start_knot = right_basis
        right_basis_end_knot = right_basis_start_knot + current_degree
        right_basis_alpha = (t - knots[right_basis_start_knot]) / (
            knots[right_basis_end_knot] - knots[right_basis_start_knot]
        )
        d[current_degree, current_degree] = (
            right_basis_alpha * d[previous_degree, previous_degree]
        )

    # Evaluate central part
    for previous_degree in range(1, p):
        current_degree = previous_degree + 1
        current_degree_basis_numbers = current_degree + 1
        left_most_basis = root_basis - current_degree
        for i in range(1, current_degree_basis_numbers - 1):
            current_basis = left_most_basis + i
            left_side_start_knot = current_basis
            left_side_end_knot = current_basis + current_degree
            left_side_alpha = (t - knots[left_side_start_knot]) / (
                knots[left_side_end_knot] - knots[left_side_start_knot]
            )
            right_side_start_knot = left_side_start_knot + 1
            right_side_end_knot = left_side_end_knot + 1
            right_side_alpha = (knots[right_side_end_knot] - t) / (
                knots[right_side_end_knot] - knots[right_side_start_knot]
            )
            d[current_degree, i] = (
                left_side_alpha * d[previous_degree, i - 1]
                + right_side_alpha * d[previous_degree, i]
            )

    return d


knot_vector = [0.0, 1.1, 2.3, 3.0, 4.5, 5.8, 6.2, 7.9, 8.4, 9.2]
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
print()
deBoorBasis_result = deBoorBasis(start_knot, x[index], knot_vector, degree)
print(
    "1, 1, 1, 1, 1",
    np.sum(deBoorBasis_result[-1, :]),
)
print(
    "0, 1, 1, 1, 1",
    np.sum(deBoorBasis_result[-1, 1:]),
)
print(
    "0, 0, 1, 1, 1",
    np.sum(deBoorBasis_result[-1, 2:]),
)
print(
    "0, 0, 0, 1, 1",
    np.sum(deBoorBasis_result[-1, 3:]),
)
print(
    "0, 0, 0, 0, 1",
    np.sum(deBoorBasis_result[-1, 4:]),
)

# Test numerical precision issue
# There is an 1e-16 error on the deBoorBasis function
print(deBoorBasis_result[-1, :] - N[:, index])
print(np.sum(deBoorBasis_result[-1, :]) - N_cum[0, index])
print(
    np.sum(deBoorBasis_result[-1, :])
    - deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 0)
)
print(
    np.sum(deBoorBasis_result[-1, 1:])
    - deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 1)
)
print(
    np.sum(deBoorBasis_result[-1, 2:])
    - deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 2)
)
print(
    np.sum(deBoorBasis_result[-1, 3:])
    - deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 3)
)
print(
    np.sum(deBoorBasis_result[-1, 4:])
    - deBoorBasisSparseV2(start_knot, x[index], knot_vector, degree, 4)
)
