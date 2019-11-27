from functools import partial
import argparse
import sys

from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm
import numpy as np

from sigtestv.stats import MeanMaxEstimator, dkde_from_sample_rvs


def make_tnorm_rvs(loc, scale, c: float = 0, d: float = 1):
    a = (c - loc) / scale
    b = (d - loc) / scale
    return partial(stats.truncnorm.rvs, loc=loc, scale=scale, a=a, b=b)


def make_texpon_rvs(loc, scale, d: float = 1):
    b = (d - loc) / scale
    return partial(stats.truncexpon.rvs, loc=loc, scale=scale, b=b)


def simulate(args, estimator_cls, gen_fn, header, ax, plot_range=False):
    y = []
    y_p25 = []
    y_p75 = []
    y_true = []
    x = list(range(1, args.subsample_size + 1))
    for k in tqdm(x):
        true_samples = []
        for _ in range(10):
            maximums = np.max(gen_fn(size=(k, 100000)), 0)
            true_samples.extend(maximums)
        true_param = np.mean(true_samples)
        y_true.append(true_param)
        mme = estimator_cls(options=dict(n=k))
        name = mme.name
        estimates = []
        for _ in range(args.num_iters):
            estimates.append(mme.estimate_point(gen_fn(size=args.sample_size)))
        y.append(np.mean(estimates))
        y_p25.append(np.quantile(estimates, 0.25))
        y_p75.append(np.quantile(estimates, 0.75))
    p = ax.plot(x, y, label=f'{name} ({header})')
    if plot_range:
        c = p[-1].get_color()
        ax.fill_between(x, y_p25, y_p75, color=c, alpha=0.25)
    ax.plot(x, y_true, label=f'True Value ({header})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', '-s', type=int, default=50)
    parser.add_argument('--subsample-size', '-k', type=int, default=None)
    parser.add_argument('--num-iters', '-n', type=int, default=5000)
    parser.add_argument('--use-kde', action='store_true')
    args = parser.parse_args()

    if args.subsample_size is None:
        args.subsample_size = args.sample_size

    if args.use_kde:
        results = list(sys.stdin)
        split_idx = results.index('\n')
        results1 = list(filter(lambda x: x.strip() != '', results[:split_idx]))
        results2 = list(filter(lambda x: x.strip() != '', results[split_idx + 1:]))
        results1 = np.array(results1, dtype=float)
        results2 = np.array(results2, dtype=float)
        gen_fn1 = dkde_from_sample_rvs(results1)
        gen_fn2 = dkde_from_sample_rvs(results2)
    else:
        gen_fn1 = make_tnorm_rvs(0.5, 0.1, d=0.6)
        gen_fn2 = make_tnorm_rvs(0.25, 0.2, d=0.75)
    plt.hist(gen_fn1(size=10000), bins=100)
    plt.show()
    plt.hist(gen_fn2(size=10000), bins=100)
    plt.show()

    fig, ax = plt.subplots()
    simulate(args, MeanMaxEstimator, gen_fn1, 'Small', ax)
    simulate(args, MeanMaxEstimator, gen_fn2, 'Big', ax)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
