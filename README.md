# MLfreq

Machine learning density functional will predict
energy with small fluctuations due to the interpolation
nature of machine learning model.

This module will calculate the vibrational frequency
from numerical energy.

energy input unit is kcal/mol
mass input unit is atomic unit (electron mass = 1)
x0 input unit is Angstrom (1e-10 meter)

# Attributes:
    energy: a function output potential energy
            with input coordinate list
    T: kinetic matrix
    x0: equilibrium position
    N: number of coordinates

# example

    >>> V = lambda x: 0.5 * 627.51 * ((x[0] - x[1]) / 0.529177249) ** 2
    >>> f = Frequency(V, mass=[1, 1], x0=[0, 0])
    >>> np.allclose(f.omega(0.1), [310383.67689266])
    True
    >>> np.allclose(f.V, np.array([[1., -1.], [-1., 1.]]))
    True
    >>> np.allclose(f.omegas([0.1, 1., 2.], 1), [310383.67689266] * 3)
    True
