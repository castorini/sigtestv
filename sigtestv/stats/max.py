from dataclasses import dataclass

import numpy as np

from .ci import bootstrap_ci, compute_ecdf_ci_bands
from .estimator import Estimator, harrelldavis_estimate
from .utils import ecdf, pos_mean_ecdf


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
        n = self.options.get('n', len(sample))
        sample = np.sort(sample)
        cdf, sample = ecdf(sample)
        return pos_mean_ecdf(cdf ** n, sample)

    def estimate_interval(self, sample, alpha=0.05):
        ci_method = self.options['ci_method']
        if ci_method == 'percentile-bootstrap':
            return bootstrap_ci(sample,
                                self.estimate_point,
                                alpha=alpha,
                                method=self.options['ci_method'],
                                ci_samples=self.options['ci_samples'])
        elif ci_method == 'direct':
            n = self.options.get('n', len(sample))
            est = self.estimate_point(sample)
            (lecdf, uecdf), sample = compute_ecdf_ci_bands(sample, alpha, k=n)
            qa1 = pos_mean_ecdf(uecdf, sample)
            qa2 = pos_mean_ecdf(lecdf, sample)
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
        n = self.options.get('n', len(sample))
        if self.options['estimate_method'] == 'harrelldavis':
            return harrelldavis_estimate(sample, q, pow=n)
        if self.options['estimate_method'] == 'direct':
            return np.quantile(sample, q ** (1 / n), interpolation='nearest')

    def estimate_interval(self, sample, alpha=0.05):
        return bootstrap_ci(sample,
                            self.estimate_point,
                            alpha=alpha,
                            method=self.options['ci_method'],
                            ci_samples=self.options['ci_samples'])
