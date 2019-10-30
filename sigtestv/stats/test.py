from dataclasses import dataclass, field
from typing import Any, Dict

from scipy import stats
import numpy as np

from .utils import compute_pr_x_ge_y


@dataclass(frozen=True)
class TwoSampleHypothesisTest(object):
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        raise NotImplementedError


@dataclass(frozen=True)
class StudentsTTest(TwoSampleHypothesisTest):

    @property
    def name(self):
        if self.options.get('unequal_var'):
            return 'Welch\'s t-test'
        else:
            return 't-test'

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        t, p = stats.ttest_ind(sample1, sample2, **self.options)
        return p / 2 < alpha and t < 0, t, p


@dataclass(frozen=True)
class MannWhitneyUTest(TwoSampleHypothesisTest):

    @property
    def name(self):
        return 'Mann-Whitney U test'

    def __post_init__(self):
        if 'alternative' not in self.options:
            self.options['alternative'] = 'less'

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        U, p = stats.mannwhitneyu(sample1, sample2, **self.options)
        return p < alpha, U, p


@dataclass(frozen=True)
class ASDTest(TwoSampleHypothesisTest):

    @property
    def name(self):
        return 'Almost Stochastic Dominance test'

    def test(self, sample1: np.ndarray, sample2: np.ndarray, alpha=0.05):
        tmp = sample2
        sample2 = sample1
        sample1 = tmp
        phi = stats.norm.ppf(alpha)
        epsilons = []
        n = len(sample1)
        m = len(sample2)
        c = np.sqrt(n * m / (n + m))
        eps_fn = lambda x, y: 1 - compute_pr_x_ge_y(x, y)
        eps_orig = eps_fn(sample1, sample2)
        for _ in range(1000):
            bs1 = np.random.choice(sample1, n)
            bs2 = np.random.choice(sample2, m)
            epsilons.append(c * (eps_fn(bs1, bs2) - eps_orig))
        min_eps = eps_orig - (1 / c) * np.std(epsilons) * phi
        return min_eps < self.options.get('threshold', 0.5), min_eps, alpha
