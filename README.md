# pass_logit

PySpark implementation of approximate Bayesian logistic regression algorithm
from

Jonathan Huggins, Ryan Adams, and Tamara Broderick (2017).
["PASS-GLM: polynomial approximate
sufficient  statistics  for  scalable  Bayesian  GLM  inference."](https://arxiv.org/pdf/1709.09216.pdf)
In _Advances in Neural Information Processing Systems_.

API:

```
pass_logit.get_approx_log_like_op(
    labeled_points=None, xs=None, ys=None, M=2, R=4
)
```

Returns Theano Op which takes parameter vector (theta) and returns approximate
log likelihood.
Computation of the Op takes O(1) time as the number of data points goes to
infinity.
The Op has a gradient method, which allows it to be used by samplers like No
U-Turn Sampler.

Parameters:

Either 
`labeled_points` is `None` or
`xs` is `None` and `ys` is `None`.
If `labeled_points` is `None`, then PySpark is not used (no Spark installation
required).

* `labeled_points`. PySpark RDD of `LabeledPoint`s where the labels are
  ±1.
* `xs`. Numpy array of shape `(n, d)` where `n` is number of data points,
  `d` is dimension of feature space.
* `ys`. Numpy array of shape `(n,)`, where each value is ±1.
* `M`. Chebyshev polynomial degree. Set to `2` or `6`. Higher values cause
  much slower running time but higher accuracy.
* `R`. Radius of interval used to determine Chebyshev polynomial.

The paper suggests that `R`=4 is appropriate for many datasets. The error
of the approximate log likelihood can be high if its argument is 
roughly `R` or greater (in absolute value).
Greater `R` values give a wider interval in which error is small but cause
greater error within the interval.

See `demo.ipynb` for a complete example.

Links
* <https://docs.pymc.io/notebooks/blackbox_external_likelihood.html>
* <https://docs.pymc.io/notebooks/GLM-logistic.html>
* <https://bitbucket.org/jhhuggins/pass-glm>

