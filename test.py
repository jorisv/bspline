import numpy as np
from scipy.interpolate import BSpline
import matplotlib.pyplot as plt
from bspline import Bspline

knot_vector = [0, 1, 2, 3, 4, 5, 6]
knot_vector = [2, 2, 2, 3, 4, 4, 4]
degree = 2
order = degree + 1
nb_control = len(knot_vector) - order
basis = Bspline(knot_vector, degree)

# control = np.random.rand(nb_control, 2)
control = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 6.0], [4.0, 0.0]])
print(nb_control)
print(control)

x, N, N_cum = basis.compute(False)


points = np.zeros((x.shape[0], 2))
for i, xi in enumerate(x):
    for j in range(nb_control):
        points[i, :] += N[j, i] * control[j, :]

points_cum = np.zeros((x.shape[0], 2))
for i, xi in enumerate(x):
    points_cum[i, :] = N_cum[0, i] * control[0, :]
    for j in range(1, nb_control):
        points_cum[i, :] += N_cum[j, i] * (control[j, :] - control[j - 1, :])


# spl = BSpline(knot_vector, control, degree, False)
# points = spl(x)
_, ax = plt.subplots()
ax.plot(points[:, 0], points[:, 1], linewidth=2, color="green")
ax.plot(points_cum[:, 0], points_cum[:, 1], linewidth=2, color="blue")
ax.plot(control[:, 0], control[:, 1], "ro")


basis.plot(x, N, N_cum)
