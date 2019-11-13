from functools import partial
import argparse

from matplotlib import pyplot as plt
from scipy import stats
from tqdm import trange
import numpy as np

from sigtestv.stats import MeanMaxEstimator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', '-s', type=int, default=50)
    parser.add_argument('--subsample-size', '-k', type=int, default=None)
    parser.add_argument('--num-iters', '-n', type=int, default=10000)
    parser.add_argument('--dataset-size', '-dsz', type=int, default=67000)
    parser.add_argument('--cache1', type=str)
    args = parser.parse_args()

    if args.subsample_size is None:
        args.subsample_size = args.sample_size

    N = args.dataset_size
    xs = np.linspace(0, 1, N)
    c = 16
    tn_cdf = partial(stats.truncnorm.cdf, scale=0.5 / c, loc=0.5, a=-c, b=c)
    probs = tn_cdf(xs + 1 / (2 * N)) - tn_cdf(xs - 1 / (2 * N))
    probs = probs / probs.sum()
    gen_fn = partial(np.random.choice, xs, p=probs)
    plt.hist(gen_fn(size=10000), bins=100)
    plt.show()

    estimator = MeanMaxEstimator(dict(n=args.subsample_size))
    gt_distn = []
    for _ in trange(args.num_iters):
        x = estimator.estimate_point(gen_fn(size=args.sample_size))
        gt_distn.append(x)
    gt_distn = np.array(gt_distn)
    mu = np.mean(gt_distn)
    plt.hist(gt_distn, bins=100)
    plt.axvline(mu, color='r')
    plt.show()

    exp_maximums_ss = []
    fig, ax = plt.subplots()
    x = gen_fn(size=int((0.5 * args.sample_size) ** 2))
    for _ in trange(args.num_iters):
        sample = np.random.choice(x, args.sample_size, replace=False)
        exp_maximums_ss.append(estimator.estimate_point(sample))
    a, b = np.quantile(exp_maximums_ss, (0.025, 0.975))
    ax.hist([gt_distn, exp_maximums_ss], bins=100, label=['Truth', 'Subsampling'])
    ax.legend()
    plt.axvline(mu, color='r')
    plt.axvline(a)
    plt.axvline(b)
    plt.show()

    fig, ax = plt.subplots()
    exp_maximums_bs = []
    x = x[:args.sample_size]
    x = gen_fn(size=args.sample_size)
    for _ in trange(args.num_iters):
        sample = np.random.choice(x, args.sample_size, replace=True)
        exp_maximums_bs.append(estimator.estimate_point(sample))
    ax.hist([gt_distn, exp_maximums_ss, exp_maximums_bs], bins=100, label=['Truth', 'Subsampling', 'Bootstrap'])
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
