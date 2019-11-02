from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from tqdm import trange
import numpy as np

from .test import TwoSampleHypothesisTest
from .utils import compute_pr_x_ge_y


@dataclass
class ResultPopulationPair(object):
    pop_x: np.ndarray
    pop_y: np.ndarray

    def compute_pr_x_ge_y(self):
        return compute_pr_x_ge_y(self.pop_x, self.pop_y)

    @property
    def ordered_pops(self):
        if self.compute_pr_x_ge_y() > 0.5:
            pop_big = self.pop_x
            pop_small = self.pop_y
        else:
            pop_small = self.pop_x
            pop_big = self.pop_y
        return pop_small, pop_big

    def simulate_statistical_power(self, n1, tests: Sequence[TwoSampleHypothesisTest], use_tqdm=False, **kwargs):
        pop_small, pop_big = self.ordered_pops
        iters = kwargs.get('iters', 500)
        alpha = kwargs.get('alpha', 0.05)
        n2 = kwargs.get('n2', n1)

        counter = Counter()
        for _ in trange(iters, disable=not use_tqdm):
            sx = np.random.choice(pop_small, n1, replace=False)
            sy = np.random.choice(pop_big, n2, replace=False)

            for test in tests:
                reject, stat, p = test.test(sx, sy, alpha=alpha)
                counter[test.name] += int(reject)
        return {k: v / iters for k, v in counter.items()}


@dataclass
class ResultPopulationSingle(object):
    pop: np.ndarray

    def simulate_type1_error(self, n, tests: Sequence[TwoSampleHypothesisTest], use_tqdm=False, **kwargs):
        iters = kwargs.get('iters', 500)
        alpha = kwargs.get('alpha', 0.05)

        counter = Counter()
        for _ in trange(iters, disable=not use_tqdm):
            for test in tests:
                sx1 = np.random.choice(self.pop, n, replace=False)
                sx2 = np.random.choice(self.pop, n, replace=False)
                reject, stat, p = test.test(sx1, sx2, alpha=alpha)
                counter[test.name] += int(reject)
        return {k: v / iters for k, v in counter.items()}
