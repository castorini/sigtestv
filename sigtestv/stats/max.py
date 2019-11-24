from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .ci import bootstrap_ci, compute_ecdf_ci_bands
from .estimator import Estimator, harrelldavis_estimate
from .utils import ecdf, pos_mean_ecdf
from .var_reduce import cv_adjust
from sigtestv.utils import id_wrap, WrappedObject


@lru_cache(maxsize=10000)
def compute_expected_rank(K, n):
    denom = np.arange(n - K + 1, n + 1)
    coeffs = np.array([np.arange(k - K + 1, k + 1) / denom for k in range(K - 1, n + 1)])
    coeffs = np.prod(coeffs, 1)
    delta = coeffs[1:] - coeffs[:-1]
    return np.sum(np.arange(K, n + 1) * delta)


def rankify(arr: np.ndarray):
    ranks = np.empty_like(arr)
    ranks[arr.argsort()] = np.arange(1, len(arr) + 1)
    return ranks


@lru_cache(maxsize=1000)
def cached_rankify(arr: WrappedObject):
    return rankify(arr.x) # TODO: generalize cached numpy routines, add typing


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
class CorrectedMeanMaxEstimator(Estimator):

    def __post_init__(self):
        if 'ci_samples' not in self.options:
            self.options['ci_samples'] = 1000
        if 'ci_method' not in self.options:
            self.options['ci_method'] = 'percentile-bootstrap'
        if 'samples' not in self.options:
            self.options['samples'] = 5000
        if 'method' not in self.options:
            self.options['method'] = 'subsample'
        if 'vr_methods' not in self.options:
            self.options['vr_methods'] = []
        if 'cv_method' not in self.options:
            self.options['cv_method'] = 'rank'

    @property
    def name(self):
        return f'{"Subsampling " if self.options["method"] == "subsample" else ""}Corrected MeanMax estimator (VR={self.options["vr_methods"]}'\
               f' {self.options["cv_method"]})'

    def estimate_point(self, sample: np.ndarray):
        n = self.options.get('n', len(sample))
        use_cv = 'cv' in self.options['vr_methods']
        use_av = 'av' in self.options['vr_methods']
        cv_rank = self.options['cv_method'] == 'rank'
        if self.options['method'] == 'mean':
            chunks = np.array_split(sample, len(sample) // n)
            chunks = [chunk[:n] for chunk in chunks]
            return np.mean([np.max(chunk) for chunk in chunks])
        elif self.options['method'] == 'subsample':
            n_samples = self.options['samples']
            indices = np.arange(len(sample))
            if use_cv:
                if cv_rank:
                    ranks = cached_rankify(id_wrap(sample))
                    expected_cv = compute_expected_rank(n, len(sample))
                else:
                    mme = MeanMaxEstimator(options=dict(n=n))
                    expected_cv = np.mean([mme.estimate_point(np.random.choice(sample, n, replace=False)) for _ in range(2000)])
            if n >= len(sample) // 2 or not use_av:
                rand_inds = [np.random.choice(indices, n, replace=False) for _ in range(n_samples)]
                samples = np.array([np.max(sample[rand_ind]) for rand_ind in rand_inds])
                if use_cv:
                    if cv_rank:
                        cv_values = np.array([np.max(ranks[rand_ind]) for rand_ind in rand_inds])
                    else:
                        cv_values = np.array([mme.estimate_point(sample[rand_ind]) for rand_ind in rand_inds])
                    samples = cv_adjust(samples, cv_values, expected_cv)
            else:
                rand_inds = [np.random.choice(indices, 2 * n, replace=False) for _ in range(n_samples)]
                samples = np.array([np.max(sample[rand_ind].reshape((2, n)), 1) for rand_ind in rand_inds])
                if use_cv:
                    if cv_rank:
                        cv_values = np.array([np.max(ranks[rand_ind].reshape((2, n)), 1) for rand_ind in rand_inds])
                    else:
                        cv_values1 = np.array([mme.estimate_point(sample[rand_ind[:n]]) for rand_ind in rand_inds])
                        cv_values2 = np.array([mme.estimate_point(sample[rand_ind[n:]]) for rand_ind in rand_inds])
                        cv_values = np.vstack((cv_values1, cv_values2)).T
                    samples[:, 0] = cv_adjust(samples[:, 0], cv_values[:, 0], expected_cv)
                    samples[:, 1] = cv_adjust(samples[:, 1], cv_values[:, 1], expected_cv)
            return np.mean(samples)

    def estimate_interval(self, sample, alpha=0.05):
        ci_method = self.options['ci_method']
        if ci_method == 'percentile-bootstrap':
            return bootstrap_ci(sample,
                                self.estimate_point,
                                alpha=alpha,
                                method=self.options['ci_method'],
                                ci_samples=self.options['ci_samples'])


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
