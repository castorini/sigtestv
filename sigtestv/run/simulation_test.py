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
    args = parser.parse_args()

    N = args.dataset_size
    xs = np.linspace(0, 1, N)
    tn_cdf = partial(stats.truncnorm.cdf, scale=0.25, loc=0.5, a=-2, b=2)
    probs = tn_cdf(xs + 1 / (2 * N)) - tn_cdf(xs - 1 / (2 * N))
    probs = probs / probs.sum()
    gen_fn = partial(np.random.choice, xs, p=probs)
    plt.hist(gen_fn(size=10000), bins=100)
    plt.show()

    gt_distn = []
    for _ in trange(args.num_iters // 100):
        x = gen_fn(size=(args.num_iters // 4, args.sample_size, 100))
        gt_distn.extend(np.mean(np.max(x, axis=1), axis=0).tolist())
    plt.hist(gt_distn, bins=100)
    plt.show()

    exp_maximums_ss = []
    fig, ax = plt.subplots()
    x = gen_fn(size=int((args.sample_size / 2) ** 2))
    for _ in trange(args.num_iters):
        samples = np.random.choice(x, (args.num_iters // 4, args.sample_size), replace=True)
        exp_maximums_ss.append(np.mean(np.max(samples, axis=1), axis=0))
    ax.hist([gt_distn, exp_maximums_ss], bins=100, label=['Truth', 'Subsampling'])
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    exp_maximums_bs = []
    x = x[:args.sample_size]
    for _ in trange(args.num_iters):
        samples = np.random.choice(x, (args.num_iters // 4, args.sample_size), replace=True)
        exp_maximums_bs.append(np.mean(np.max(samples, axis=1), axis=0))
    ax.hist([gt_distn, exp_maximums_ss, exp_maximums_bs], bins=100, label=['Truth', 'Subsampling', 'Bootstrap'])
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
