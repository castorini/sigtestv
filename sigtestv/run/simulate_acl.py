from functools import partial
import argparse
import sys

from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm, trange
import numpy as np

from sigtestv.stats import MeanMaxEstimator, dkde_from_sample_rvs


def make_tnorm_rvs(loc, scale, c: float = 0, d: float = 1):
    a = (c - loc) / scale
    b = (d - loc) / scale
    return partial(stats.truncnorm.rvs, loc=loc, scale=scale, a=a, b=b)


def make_texpon_rvs(loc, scale, d: float = 1):
    b = (d - loc) / scale
    return partial(stats.truncexpon.rvs, loc=loc, scale=scale, b=b)


def simulate_bs(args, estimator_cls, gen_fn, header):
    k = args.subsample_size
    mme = estimator_cls(options=dict(n=k))
    true_samples = []
    for _ in range(10):
        maximums = np.max(gen_fn(size=(k, 100000)), 0)
        true_samples.extend(maximums)
    true_param = np.mean(true_samples)
    results = []
    for _ in trange(args.num_iters):
        _, (l, u) = mme.estimate_interval(gen_fn(size=args.sample_size))
        results.append(int(l <= true_param <= u))
    tqdm.write(f'{header} (k={k}): {100 * np.mean(results):.2f}')


def simulate_mme_test(args, estimator_cls, gen_fn1, gen_fn2):
    for n in trange(1, args.sample_size + 1):
        true_samples = []
        for _ in range(10):
            maximums = np.max(gen_fn1(size=(n, 100000)), 0)
            true_samples.extend(maximums)
        true_param1 = np.mean(true_samples)
        true_samples = []
        for _ in range(10):
            maximums = np.max(gen_fn2(size=(n, 100000)), 0)
            true_samples.extend(maximums)
        true_param2 = np.mean(true_samples)

        mme = estimator_cls(options=dict(n=n))
        name = mme.name
        errors = []
        if args.correct_bias:
            bias1 = np.mean([mme.estimate_point(gen_fn1(size=args.sample_size)) for _ in range(args.num_iters)]) - true_param1
            bias2 = np.mean([mme.estimate_point(gen_fn2(size=args.sample_size)) for _ in range(args.num_iters)]) - true_param2
        for _ in range(args.num_iters):
            estimate1 = mme.estimate_point(gen_fn1(size=args.sample_size))
            estimate2 = mme.estimate_point(gen_fn2(size=args.sample_size))
            if args.correct_bias:
                estimate1 -= bias1
                estimate2 -= bias2
            error_flag = int((estimate1 < estimate2 and true_param1 > true_param2) or \
                             (estimate1 > estimate2 and true_param1 < true_param2))
            errors.append(error_flag)
        error_rate = 100 * np.mean(errors)
        tqdm.write(f'{name} (n={n}) {error_rate:.2f}')


def simulate_mme(args, estimator_cls, gen_fn, header, ax, plot_range=False):
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
    parser.add_argument('--action', '-a', type=str, default='mme', choices=['mme', 'bs', 'mme-test'])
    parser.add_argument('--show-hist', '-sh', action='store_true')
    parser.add_argument('--correct-bias', '-cb', action='store_true')
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

    if args.show_hist:
        plt.hist(gen_fn1(size=10000), bins=100)
        plt.show()
        plt.hist(gen_fn2(size=10000), bins=100)
        plt.show()

    if args.action == 'mme':
        fig, ax = plt.subplots()
        simulate_mme(args, MeanMaxEstimator, gen_fn1, 'Small', ax)
        simulate_mme(args, MeanMaxEstimator, gen_fn2, 'Big', ax)
        plt.legend()
        plt.show()
    elif args.action == 'bs':
        simulate_bs(args, MeanMaxEstimator, gen_fn1, 'Small')
        simulate_bs(args, MeanMaxEstimator, gen_fn2, 'Big')
    elif args.action == 'mme-test':
        simulate_mme_test(args, MeanMaxEstimator, gen_fn1, gen_fn2)


if __name__ == '__main__':
    main()
