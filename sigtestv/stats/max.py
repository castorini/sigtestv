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
            self.options['ci_samples'] = 5000
        if 'ci_method' not in self.options:
            self.options['ci_method'] = 'percentile-bootstrap'

    @property
    def name(self):
        return 'MeanMax estimator'

    def estimate_point(self, sample: np.ndarray):
        n = self.options.get('n', len(sample))
        if not self.options.get('sorted'):
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


@lru_cache(maxsize=100000)
def cached_le_prob(sample: WrappedObject, value):
    return np.mean(sample.x < value)


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
        if 'output_prob' not in self.options:
            self.options['output_prob'] = False

    @property
    def name(self):
        return f'{"Subsampling " if self.options["method"] == "subsample" else ""}Corrected MeanMax estimator (VR={self.options["vr_methods"]}'\
               f' {self.options["cv_method"]})'

    def estimate_point(self, sample: np.ndarray):
        n = self.options.get('n', len(sample))
        use_cv = 'cv' in self.options['vr_methods']
        use_av = 'av' in self.options['vr_methods']
        cv_rank = self.options['cv_method'] == 'rank'
        cdf_map = ecdf(sample, sorted=False, equality=False, return_map=True)
        if self.options['method'] == 'mean':
            chunks = np.array_split(sample, len(sample) // n)
            chunks = [chunk[:n] for chunk in chunks]
            return np.mean([np.max(chunk) for chunk in chunks])
        elif self.options['method'] == 'subsample':
            n_samples = self.options['samples']
            samples = np.empty(n_samples)
            output_prob = self.options['output_prob']
            if output_prob:
                probs = np.empty(n_samples)
            for idx in range(n_samples):
                samples[idx] = tmax = np.max(np.random.choice(sample, n, replace=False))
                if output_prob:
                    probs[idx] = cdf_map[tmax]
            if output_prob:
                return np.mean(samples), np.mean(probs)
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


@dataclass(frozen=True)
class MeanMaxBudgetEstimator(Estimator):

    def __post_init__(self):
        if 'alpha' not in self.options:
            self.options['alpha'] = 0.95
        if 'budget' not in self.options:
            self.options['budget'] = 1
        if 'method' not in self.options:
            self.options['method'] = 'forward'
        if 'sorted' not in self.options:
            self.options['sorted'] = False
        if 'theta_method' not in self.options:
            self.options['theta_method'] = 'mme'

    @property
    def name(self):
        return 'MeanMax Budget Estimator'

    def estimate_point(self, sample: np.ndarray):
        k = self.options.get('n', len(sample))
        alpha = self.options['alpha']
        method = self.options['method']
        theta_method = self.options['theta_method']
        le_prob = None
        if theta_method == 'mme':
            mme = MeanMaxEstimator(options=dict(n=k))
        else:
            mme = CorrectedMeanMaxEstimator(options=dict(n=k, output_prob=True))
        try:
            thetahat, le_prob = mme.estimate_point(sample)
        except:
            thetahat = mme.estimate_point(sample)
        if method == 'forward':
            num = np.log(1 - alpha)
            if le_prob is None: le_prob = np.mean(sample <= thetahat)
            denom = np.log(le_prob)
            return (np.ceil(num / denom) if self.options.get('ceil', True) else num / denom) * self.options['budget']
        elif method == 'backward':
            for idx in range(len(sample) ** 2):
                qme = QuantileMaxEstimator(options=dict(n=idx + 1, quantile=1 - alpha))
                x = qme.estimate_point(sample)
                if x >= thetahat:
                    break
            return idx + 1


ForwardEstimator = MeanMaxBudgetEstimator.gen_class(dict(method='forward'))
BackwardEstimator = MeanMaxBudgetEstimator.gen_class(dict(method='backward'))
