from dataclasses import dataclass
from functools import lru_cache

from scipy.special import beta, betainc
import numpy as np

from .estimator import Estimator, harrelldavis_estimate


@lru_cache(maxsize=1000)
def quantile_max_coeffs(n, q, linspace_n=None):
    def Fn(x):
        return alpha * betainc(k, n + 1 - k, np.power(x, 1 / n)) * beta(k, n + 1 - k)
    if linspace_n is None:
        linspace_n = n
    k = (n + 1) * q
    b = (n + 1) * (1 - q)
    alpha = 1 / beta(k, b)
    arr = np.linspace(0, 1, linspace_n + 1)
    arr = Fn(arr)
    arr[0] = 0
    return arr


@dataclass(frozen=True)
class QuantileMaxEstimator(Estimator):

    def __post_init__(self):
        if 'quantile' not in self.options:
            self.options['quantile'] = 0.5
        if 'ci_method' not in self.options:
            self.options['ci_method'] = 'percentile-bootstrap'
        if 'ci_samples' not in self.options:
            self.options['ci_samples'] = 1000
        if 'estimate_method' not in self.options:
            self.options['estimate_method'] = 'harrelldavis'

    @property
    def name(self):
        return f'QuantileMax ({self.options["quantile"]}) {self.options["estimate_method"]} estimator'

    def estimate_point(self, sample: np.ndarray):
        q = self.options['quantile']
        n = len(sample)
        if self.options['estimate_method'] == 'harrelldavis':
            return harrelldavis_estimate(sample, q, pow=n)

        return

    def estimate_interval(self, sample, alpha=0.05):
        raise NotImplementedError


print(QuantileMaxEstimator(dict(quantile=0.05)).estimate_point(np.repeat(np.arange(1, 100), 10)))
# n = 8
# q = 0.5
# print(quantile_max_coeffs(n, q))
# a = (n + 1) * q
# b = (n + 1) * (1 - q)
# print(bt(a, b).cdf(np.linspace(0, 1, n)))