from functools import partial
import argparse

from matplotlib import pyplot as plt
from scipy import stats
from tqdm import trange
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', '-s', type=int, default=50)
    parser.add_argument('--num-iters', '-n', type=int, default=10000)
    parser.add_argument('--dataset-size', '-dsz', type=int, default=67000)
    parser.add_argument('--cache1', type=str)
    args = parser.parse_args()

    N = args.dataset_size
    xs = np.linspace(0, 1, N)
    c = 16
    tn_cdf = partial(stats.truncnorm.cdf, scale=0.5 / c, loc=0.5, a=-c, b=c)
    probs = tn_cdf(xs + 1 / (2 * N)) - tn_cdf(xs - 1 / (2 * N))
    probs = probs / probs.sum()
    gen_fn = partial(np.random.choice, xs, p=probs)
    plt.hist(gen_fn(size=10000), bins=100)
    plt.show()

    gt_distn = []
    bypass = False
    if args.cache1:
        try:
            gt_distn = np.load(args.cache1)
            bypass = True
        except FileNotFoundError:
            pass
    if not bypass:
        for _ in trange(args.num_iters // 100):
            x = gen_fn(size=(args.num_iters, args.sample_size, 100))
            gt_distn.extend(np.mean(np.max(x, axis=1), axis=0).tolist())
    gt_distn = np.array(gt_distn)
    if args.cache1:
        np.save(args.cache1, gt_distn)
    mu = np.mean(gt_distn)
    plt.hist(gt_distn, bins=100)
    plt.axvline(mu, color='r')
    plt.show()

    exp_maximums_ss = []
    fig, ax = plt.subplots()
    x = gen_fn(size=int(10 * args.sample_size))
    for _ in trange(args.num_iters // 100):
        samples = np.random.choice(x, (args.num_iters, args.sample_size, 100), replace=True)
        exp_maximums_ss.extend(np.mean(np.max(samples, axis=1), axis=0))
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
    for _ in trange(args.num_iters // 100):
        samples = np.random.choice(x, (args.num_iters, args.sample_size, 100), replace=True)
        exp_maximums_bs.extend(np.mean(np.max(samples, axis=1), axis=0))
    ax.hist([gt_distn, exp_maximums_ss, exp_maximums_bs], bins=100, label=['Truth', 'Subsampling', 'Bootstrap'])
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
