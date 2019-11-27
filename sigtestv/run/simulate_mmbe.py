from collections import defaultdict
from functools import partial, lru_cache
import argparse
import sys

from matplotlib import pyplot as plt
from tqdm import trange, tqdm
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

from sigtestv.stats import ForwardEstimator, BackwardEstimator, dfromc_rvs, MeanMaxEstimator, WrappedObject, id_wrap, ecdf


@lru_cache(maxsize=100000)
def cached_point_estimate(wrapped_pop: WrappedObject, n):
    return MeanMaxEstimator(options=dict(n=n, sorted=True)).estimate_point(wrapped_pop.x)


def main():
    def print_idx(data, idx, header):
        tqdm.write(f'{header:<20} Bias: {data["bias"][idx]:<8.4f} Var: {data["var"][idx]:<8.4f} MSE: {data["mse"][idx]:<8.4f} %<: {100 * data["pless"][idx]:<8.6f}')
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', '-ns', required=True, type=int)
    parser.add_argument('--num-iters', '-ni', required=True, type=int)
    # parser.add_argument('--dataset-size', '-dsz', type=int, required=True)
    parser.add_argument('--begin-index', '-bi', type=int, default=0)
    parser.add_argument('--use-kde', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.95)
    parser.add_argument('--step-size', '-ss', type=int, default=1)
    # parser.add_argument('--power', '-k', type=float, nargs=2)
    args = parser.parse_args()

    alpha = args.alpha
    metric_data = (defaultdict(list), defaultdict(list))
    if args.use_kde:
        results = np.array(list(sys.stdin), dtype=float)
        kde = sm.nonparametric.KDEUnivariate(results)
        kde.fit()
        pmf = kde.cdf[1:] - kde.cdf[:-1]
        pmf = pmf / pmf.sum()
        rvs_gen = partial(np.random.choice, kde.support[:-1], p=pmf)
    else:
        rvs_gen = partial(stats.uniform.rvs)
        # rvs_gen = partial(np.random.choice, np.arange(2))
        # rvs_gen = partial(stats.truncnorm.rvs, loc=0.5, a=-3, b=2)
        # rvs_gen = partial(stats.norm.rvs, loc=0.5)
    print('Sampling population...', file=sys.stderr)
    pop1 = np.sort(rvs_gen(size=2000000))
    pop2 = np.sort(rvs_gen(size=2000000))
    print('Plotting...', file=sys.stderr)
    wpop = id_wrap(pop1)
    plt.hist(rvs_gen(size=50000), bins=1000)
    plt.show()
    for n in trange(args.begin_index + 1, args.num_samples + 1, args.step_size):
        true_param = ForwardEstimator(options=dict(n=n, alpha=alpha, sorted=True, ceil=False)).estimate_point(pop1)
        true_val = MeanMaxEstimator(options=dict(n=true_param, sorted=True)).estimate_point(pop2)
        fes = []
        bes = []
        fes_true = []
        bes_true = []
        for _ in trange(args.num_iters, position=1):
            sample = rvs_gen(size=args.num_samples)
            fe = ForwardEstimator(options=dict(n=n, alpha=alpha))
            be = BackwardEstimator(options=dict(n=n, alpha=alpha))
            fe_estimate = fe.estimate_point(sample)
            be_estimate = be.estimate_point(sample)
            fes.append(fe_estimate)
            bes.append(be_estimate)
            fes_true.append(cached_point_estimate(wpop, int(fe_estimate)))
            bes_true.append(cached_point_estimate(wpop, int(be_estimate)))
        print()
        print(true_val, fes_true[-1], bes_true[-1])
        fes = np.array(fes)
        bes = np.array(bes)
        tqdm.write(str(true_param))
        metric_data[0]['bias'].append(np.mean(fes - true_param))
        metric_data[1]['bias'].append(np.mean(bes - true_param))
        metric_data[0]['pless'].append(np.mean(fes_true < true_val))
        metric_data[1]['pless'].append(np.mean(bes_true < true_val))
        metric_data[0]['mse'].append(np.mean((fes - true_param) ** 2))
        metric_data[1]['mse'].append(np.mean((bes - true_param) ** 2))
        metric_data[0]['var'].append(np.var(fes))
        metric_data[1]['var'].append(np.var(bes))
        print_idx(metric_data[0], -1, 'Forward')
        print_idx(metric_data[1], -1, 'Backward')


if __name__ == '__main__':
    main()
