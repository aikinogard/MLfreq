from itertools import combinations
import numpy as np
from scipy.linalg import eigh


class Frequency:
    """
        Machine learning density functional will predict
        energy with small fluctuations due to the interpolation
        nature of machine learning model.

        This module will calculate the vibrational frequency
        from numerical energy.

        energy input unit is kcal/mol
        mass input unit is atomic unit (electron mass = 1)
        x0 input unit is Angstrom (1e-10 meter)

        Attributes:
            energy: a function output potential energy
                    with input coordinate list
            T: kinetic matrix
            x0: equilibrium position
            N: number of coordinates

        >>> V = lambda x: 0.5 * 627.51 * ((x[0] - x[1]) / 0.529177249) ** 2
        >>> f = Frequency(V, mass=[1, 1], x0=[0, 0])
        >>> np.allclose(f.omega(0.1), [310383.67689266])
        True
        >>> np.allclose(f.V, np.array([[1., -1.], [-1., 1.]]))
        True
        >>> np.allclose(f.omegas([0.1, 1., 2.], 1), [310383.67689266] * 3)
        True
    """

    def __init__(self, energy, mass, x0):
        self.energy = energy
        self.T = np.diag(np.array(mass, "d"))
        self.x0 = np.array(x0, "d")
        self.V0 = energy(x0)
        self.N = len(x0)

    def omega(self, rc):
        """
            from hartree to cm^-1
        """
        u, v = self.eigh(rc)
        return np.sqrt(u[u > 0]) / 4.55634e-6

    def omegas(self, rcs, n=1):
        """
            return first n omega of different rcs
        """
        rec = np.empty((len(rcs), n))
        for i, rc in enumerate(rcs):
            tmp = self.omega(rc)[::-1]
            rec[i] = tmp[:n]
        return rec

    def eigh(self, rc):
        self.compute_V(rc)
        u, v = eigh(self.V, self.T)
        return u, v

    def compute_V(self, rc):
        self.k_diag = np.empty(self.N)
        for i in range(self.N):
            self.k_diag[i] = self.get_k_diag(rc, i)
        V = np.diag(self.k_diag)
        for i, j in combinations(range(self.N), 2):
            tmp = self.get_k(rc, i, j) / 2
            V[i, j] = V[j, i] = tmp
        self.V = V

    def get_k_diag(self, rc, i):
        ls = np.linspace(-rc / 2., rc / 2., 20)
        res = np.empty(len(ls))
        for m in range(len(ls)):
            lv = self.x0.copy()
            lv[i] += ls[m]
            res[m] = self.energy(lv)
        # fit 1/2 * k(r - r_0)^2 + E[r_0] = E[r]
        x = 0.5 * (ls / 0.529177249) ** 2
        # in bohr^2
        y = (res - self.V0) / 627.51
        # in hartree
        k = np.linalg.lstsq(x.reshape(-1, 1), y)[0]
        # is hartree / bohr^2
        return k

    def get_k(self, rc, i, j):
        ls = np.linspace(-rc / 2., rc / 2., 20)
        y = np.empty(len(ls) ** 2)
        x = np.empty(len(ls) ** 2)
        p = 0
        for m in range(len(ls)):
            for n in range(len(ls)):
                lv = self.x0.copy()
                lv[i] += ls[m]
                lv[j] += ls[n]
                sq = 0.5 * self.k_diag[i] * (ls[m] / 0.529177249) ** 2\
                    + 0.5 * self.k_diag[j] * (ls[n] / 0.529177249) ** 2
                # is in hartree
                y[p] = (self.energy(lv) - self.V0) / 627.51 - sq
                # is in hartree
                x[p] = 0.5 * (ls[m] / 0.529177249) * (ls[n] / 0.529177249)
                # in bohr^2
                p += 1
        k = np.linalg.lstsq(x.reshape(-1, 1), y)[0]
        return k


if __name__ == '__main__':
    import doctest
    doctest.testmod()
