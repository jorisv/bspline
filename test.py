from bspline import Bspline

# knot_vector = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
knot_vector = [0, 1, 2, 3, 4, 5, 6, 6, 6]
degree = 2
order = degree + 1
nb_control = len(knot_vector) - order
print(nb_control)
basis = Bspline(knot_vector, degree)

basis.plot()
