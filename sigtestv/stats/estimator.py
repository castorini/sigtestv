from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict

from scipy.stats import beta
import numpy as np

from sigtestv.utils import id_wrap


@dataclass(frozen=True)
class Estimator(object):
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def estimate_point(self, sample: np.ndarray):
        raise NotImplementedError

    def estimate_interval(self, sample: np.ndarray, alpha=0.05):
        raise NotImplementedError


@lru_cache(maxsize=1000)
def beta_cdf_linspace(a, b, n, pow=1):
    inds = np.linspace(0, 1, n + 1) ** pow
    cdfs = beta(a, b).cdf(inds)
    return cdfs[1:] - cdfs[:-1]


def harrelldavis_estimate(sample, q, pow=1):
    n = len(sample)
    a = (n + 1) * q
    b = (n + 1) * (1 - q)
    return np.sum(beta_cdf_linspace(a, b, n, pow=pow) * np.sort(sample))


@lru_cache(maxsize=1000)
def sorted_cache(a):
    return np.sort(a.x)


@dataclass(frozen=True)
class QuantileEstimator(Estimator):

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
        return f'Quantile ({self.options["quantile"]}) {self.options["estimate_method"]} estimator'

    def estimate_point(self, sample: np.ndarray):
        q = self.options['quantile']
        if self.options['estimate_method'] == 'harrelldavis':
            return harrelldavis_estimate(sample, q)
        elif self.options['estimate_method'] == 'direct':
            return np.quantile(sample, q)

    def estimate_interval(self, sample, alpha=0.05):
        est = self.estimate_point(sample)
        b = self.options['ci_samples']
        bs_estimates = [np.random.choice(sample, len(sample)) for _ in range(b)]
        qa1, qa2 = np.quantile(bs_estimates, (alpha / 2, 1 - alpha / 2))
        return est, (qa1, qa2)
