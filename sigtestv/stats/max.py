from dataclasses import dataclass

from scipy import stats
import numpy as np

from .estimator import Estimator, harrelldavis_estimate
from .utils import ecdf


@dataclass(frozen=True)
class MeanMaxEstimator(Estimator):

    def __post_init__(self):
        if 'ci_samples' not in self.options:
            self.options['ci_samples'] = 1000
        if 'ci_method' not in self.options:
            self.options['ci_method'] = 'percentile-bootstrap'

    @property
    def name(self):
        return 'MeanMax estimator'

    def estimate_point(self, sample: np.ndarray):
        n = len(sample)
        sample = np.sort(sample)
        cdf, _ = ecdf(sample)
        less_cdf, sample = ecdf(sample, equality=False)
        return np.sum(sample * (cdf ** n - less_cdf ** n))

    def estimate_interval(self, sample, alpha=0.05):
        est = self.estimate_point(sample)
        b = self.options['ci_samples']
        bs_estimates = [self.estimate_point(np.random.choice(sample, len(sample))) for _ in range(b)]
        if self.options['ci_method'] == 'percentile-bootstrap':
            qa1, qa2 = np.quantile(bs_estimates, (alpha / 2, 1 - alpha / 2))
        elif self.options['ci_method'] == 'reverse-bootstrap':
            qa1, qa2 = np.quantile(bs_estimates, (alpha / 2, 1 - alpha / 2))
            tmp = qa1
            qa1 = 2 * est - qa2
            qa2 = 2 * est - tmp
        return est, (qa1, qa2)


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
        if self.options['estimate_method'] == 'direct':
            return np.quantile(sample, q ** (1 / n))

    def estimate_interval(self, sample, alpha=0.05):
        est = self.estimate_point(sample)
        b = self.options['ci_samples']
        bs_estimates = [self.estimate_point(np.random.choice(sample, len(sample))) for _ in range(b)]
        qa1, qa2 = np.quantile(bs_estimates, (alpha / 2, 1 - alpha / 2))
        return est, (qa1, qa2)
