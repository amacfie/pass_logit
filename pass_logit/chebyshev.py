'''
uses code from

https://bitbucket.org/jhhuggins/pass-glm

License: MIT License (MIT)

Author: Jonathan Huggins
'''

import math
import numpy as np
from numpy.polynomial import chebyshev, Chebyshev, Polynomial
from scipy.integrate import quad

# the basis is already normalized
CHEBYSHEV_BASIS = []

def chebyshev_basis(k):
    for i in range(len(CHEBYSHEV_BASIS), k+1):
        coeffs = np.zeros(i+1)
        coeffs[-1] = (1. + np.sign(i)) / np.pi
        CHEBYSHEV_BASIS.append(Chebyshev(coeffs))
    return CHEBYSHEV_BASIS[k]


def chebyshev_bases(k):
    chebyshev_basis(k)
    return CHEBYSHEV_BASIS[:k+1]


def chebyshev_approximation(fun, degree, R=4.0):
    bases = chebyshev_bases(degree)
    approx_coeffs = []
    for i in range(len(bases)):
        approx_coeffs.append(
            quad(lambda x: fun(R*x) * bases[i](x) / np.sqrt(1 - x**2),
                    -1, 1)[0])
    std_coefs = Chebyshev(approx_coeffs).convert(kind=Polynomial).coef
    return np.power(1/R, range(degree+1)) * std_coefs

def chebyshev_approximation_alt(func, deg, a=-4, b=4):
    return Chebyshev.interpolate(
        func, deg, domain=[a,b]
    ).convert(kind=Polynomial).coef
