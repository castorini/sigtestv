import argparse

from matplotlib import pyplot as plt
from tqdm import trange
import numpy as np
import scipy.stats as stats

from sigtestv.stats import ecdf, compute_ks_distance, dfromc_rvs, compute_minimum_sample_power, maximum_cdf_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', '-n', required=True, type=int)
    parser.add_argument('--dataset-size', '-dsz', type=int, required=True)
    parser.add_argument('--power', '-k', type=float, nargs=2)
    args = parser.parse_args()

    if args.power is None:
        args.power = (1, args.num_samples + 1)
    c = 4
    cdf_kwargs = dict(scale=0.25, loc=0.5, a=-c, b=c)
    gen = dfromc_rvs(args.dataset_size, stats.truncnorm.cdf, **cdf_kwargs)
    rvs = np.sort(gen(size=10000000))
    plt.hist(rvs, bins=min(args.dataset_size, 1000))
    plt.show()
    cdf1, sample1 = ecdf(rvs)
    rvs = np.sort(gen(size=args.num_samples))
    cdf2, sample2 = ecdf(rvs)
    print(max(sample1), max(sample2))
    print(compute_minimum_sample_power(1 - cdf1[-2]))
    print(maximum_cdf_error(cdf1, sample1, cdf2, sample2))
    eps = None
    y = []
    for power in trange(*args.power):
        gt_cdf = cdf1 ** power
        sm_cdf = cdf2 ** power
        val, dist = compute_ks_distance(gt_cdf, sample1, sm_cdf, sample2)
        if eps is None: eps = dist
        y.append(dist)
    plt.plot(np.arange(*args.power), y)
    plt.show()


if __name__ == '__main__':
    main()
