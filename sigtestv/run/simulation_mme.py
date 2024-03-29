from collections import defaultdict
from functools import partial
import argparse

from matplotlib import pyplot as plt
from scipy import stats
from tqdm import trange
import numpy as np

from sigtestv.stats import MeanMaxEstimator, dfromc_rvs, compute_minimum_sample_power, CorrectedMeanMaxEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', '-s', type=int, default=50)
    parser.add_argument('--subsample-size', '-k', type=int, default=None)
    parser.add_argument('--num-iters', '-n', type=int, default=10000)
    parser.add_argument('--dataset-size', '-dsz', type=int, default=67000)
    parser.add_argument('--action', '-a', type=str, default='trajectory', choices=['trajectory', 'mse', 'ci'])
    parser.add_argument('--distribution-type', '-dtype', type=str, default='d', choices=['c', 'd', 'continuous', 'discrete'])
    args = parser.parse_args()

    if args.subsample_size is None:
        args.subsample_size = args.sample_size

    if args.distribution_type.startswith('d'):
        c = 8
        cdf_kwargs = dict(scale=0.25 / c, loc=0.5, a=-c, b=c)
        gen_fn = dfromc_rvs(args.dataset_size, stats.truncnorm.cdf, **cdf_kwargs)
        # gen_fn = dfromc_rvs(args.dataset_size, stats.uniform.cdf)
    else:
        gen_fn = partial(stats.norm.rvs, loc=0, scale=1)
        gen_fn = partial(stats.uniform.rvs)
        gen_fn = partial(stats.truncnorm.rvs, a=-3, b=5, scale=0.2, loc=0.5)

    large_N = 10000000
    pop = gen_fn(size=large_N)
    estimators = [MeanMaxEstimator(dict(n=args.subsample_size)),
                  CorrectedMeanMaxEstimator(dict(n=args.subsample_size, method='subsample')),
                  CorrectedMeanMaxEstimator(dict(n=args.subsample_size, method='mean'))]
    true_parameter = CorrectedMeanMaxEstimator(dict(n=args.subsample_size, method='mean')).estimate_point(pop)
    plt.hist(gen_fn(size=100000), bins=args.dataset_size)
    plt.show()
    fig, ax = plt.subplots()
    if args.action == 'trajectory':
        for _ in trange(args.num_iters):
            est_data = defaultdict(list)
            sample = gen_fn(size=args.sample_size)
            for n in trange(args.subsample_size, position=1):
                n += 1
                for estimator in estimators:
                    estimator.options['n'] = n
                    est_data[estimator.name].append(estimator.estimate_point(sample))
            for name, y in est_data.items():
                ax.plot(np.arange(args.subsample_size) + 1, y, label=name)
        ax.axhline(true_parameter)
        plt.legend()
        plt.show()
    elif args.action == 'ci':
        y = []
        for _ in trange(args.num_iters):
            sample = gen_fn(size=args.sample_size)
            _, (a, b) = MeanMaxEstimator(dict(n=args.subsample_size)).estimate_interval(sample)
            y.append(int(a <= true_parameter <= b))
        print(np.mean(y))
    elif args.action == 'mse':
        est_data = defaultdict(list)
        for _ in trange(args.num_iters):
            sample = gen_fn(size=args.sample_size)
            for estimator in estimators:
                est_data[estimator.name].append(estimator.estimate_point(sample))
        print('Bias')
        for name, estimates in est_data.items():
            print(name, np.mean(estimates) - true_parameter)
        print('Variance')
        for name, estimates in est_data.items():
            print(name, np.var(estimates))
        print('MSE')
        for name, estimates in est_data.items():
            print(name, np.mean((np.array(estimates) - true_parameter)**2))


if __name__ == '__main__':
    main()
