from pass_logit.chebyshev import chebyshev_approximation
from pass_logit import compositions

from pyspark import RDD
from pyspark.mllib.regression import LabeledPoint
# https://docs.sympy.org/latest/modules/ntheory.html#sympy.ntheory.multinomial.multinomial_coefficients
from sympy.ntheory.multinomial import multinomial_coefficients
import numpy as np
import theano
import theano.tensor as tt


# $\bm{t}$
def get_suff_stats(
    monomial_exponents: list,
    data_format: str,
    labeled_points=None, xs=None, ys=None
):
    if data_format == 'Spark':
        def t(x, y):
            return tuple(
                np.prod(np.power(y * x, monomial_exp))
                for monomial_exp in monomial_exponents
            )
        tns = labeled_points.map(lambda point: t(point.features,point.label))
        return tns.fold([0.0], np.add)
    elif data_format == 'Python':
        return np.sum([
            [
                # `xs` needs to be Numpy array or this will do wrong operation
                np.prod(np.power(y * x, monomial_exp))
                for monomial_exp in monomial_exponents
            ]
            for x, y in zip(xs, ys)
        ], axis=0)

def get_approx_log_like_op(
    labeled_points=None, xs=None, ys=None, M: int=2, R: int=4
):
    j = int((M - 2) / 4)
    if not j >= 0:
        raise ValueError('`M` must be 2 plus a non-negative multiple of 4.')

    if labeled_points is not None:
        data_format = 'Spark'
    elif xs is not None and ys is not None:
        data_format = 'Python'
    else:
        raise ValueError('Data not supplied.')

    if data_format == 'Spark':
        d = len(labeled_points.take(1).features)
    elif data_format == 'Python':
        d = len(xs[0])

    monomial_exponents = get_monomial_exponents(d, M)
    acoefs = get_acoefs(d, M, monomial_exponents, R)

    suff_stats = get_suff_stats(
        monomial_exponents, data_format, labeled_points, xs, ys)

    prefactor = suff_stats * acoefs

    theta = tt.dvector('theta')

    theta_ks = tt.prod((theta**tt.as_tensor(monomial_exponents)), axis=1)

    loglike_var = tt.dot(theta_ks, prefactor)

    # gradient automatically generated; see Example 3 in
    # http://deeplearning.net/software/theano/library/compile/opfromgraph.html
    return theano.OpFromGraph([theta], [loglike_var])

def get_monomial_exponents(d, M):
    # values of $\bm{k}$
    return sum(
        [
            # parts equal to zero are allowed
            list(compositions.all(n, d))
            for n in range(M+1)
        ],
        []
    )

def get_acoefs(d, M, monomial_exponents, R):
    logit = lambda x: -np.log1p(np.exp(-x))
    bMs = chebyshev_approximation(logit, M, R)

    quick_multinomial_coef = dict()
    for n in range(M+1):
        quick_multinomial_coef.update(multinomial_coefficients(d, n))

    acoefs = []
    for monomial_exp in monomial_exponents:
        # could be saved previously rather than recomputed here
        m = sum(monomial_exp)
        acoefs.append(quick_multinomial_coef[monomial_exp] * bMs[m])

    return acoefs
