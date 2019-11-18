from pathlib import Path
import argparse

from scipy import stats
from matplotlib import pyplot as plt
import numpy as np

from sigtestv.stats import QuantileMaxEstimator, MeanMaxEstimator, harrelldavis_estimate, pos_mean_ecdf, ecdf, compute_ecdf_ci_bands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-file', '-d', type=Path, required=True)
    parser.add_argument('--column', '-c', type=str, required=True)
    # parser.add_argument('--')


if __name__ == '__main__':
    main()

    # sample = [1, 1, 1, 2, 3, 4, 4, 4, 5, 6, 7, 15, 20, 30, 40, 50, 80, 90, 92, 92, 93, 93, 94, 94, 95, 95]#, 98]
    fig, ax = plt.subplots()
    for _ in range(1):
        sample = np.clip(stats.norm.rvs(loc=80, scale=2, size=100), 0, np.inf)
        # sample = np.array(sample)
        # sample = np.concatenate((sample, sample + stats.binom.rvs(5, 0.5, size=len(sample))))
        # sample = np.concatenate((sample, sample + stats.binom.rvs(5, 0.5, size=len(sample))))
        sample = np.sort(sample)
        cdf, _ = ecdf(sample)
        x = np.linspace(70, 90, 1000)
        gt = stats.norm.cdf(x, loc=80, scale=2)
        k = 10
        cdf = cdf ** k
        (lecdf, uecdf), sample = compute_ecdf_ci_bands(sample, 0.01, k=k)
        ax.step(sample, lecdf)
        ax.step(sample, cdf)
        ax.step(sample, uecdf)
        ax.step(x, gt ** k)
        q = 0.1
        lines = ([], [], [], [], [])
        N = 8
        # for n in range(N):
        #     n += 1
        #     qme = QuantileMaxEstimator(dict(quantile=q, n=n, estimate_method='harrelldavis'))
        #     qme2 = QuantileMaxEstimator(dict(quantile=q, n=n, estimate_method='direct'))
        #     mme = MeanMaxEstimator(dict(quantile=q, n=n))
        #     mme2 = MeanMaxEstimator(dict(quantile=q, n=n, ci_method='percentile-bootstrap'))
        #     est, (qa1, qa2) = mme.estimate_interval(sample, alpha=0.05)
        #     _, (qa11, qa12) = mme2.estimate_interval(sample, alpha=0.05)
        #     lines[0].append(est)
        #     lines[1].append(qa1)
        #     lines[2].append(qa2)
        #     lines[3].append(qa11)
        #     lines[4].append(qa12)
        #
        # x = np.arange(1, N + 1)
        # ax.plot(x, lines[0])
        # ax.fill_between(x, lines[1], lines[2], alpha=0.1, color='red')
        # ax.fill_between(x, lines[3], lines[4], alpha=0.1, color='orange')
    # plt.plot(np.vstack((x, x, x)).T, np.array(lines).T)
    # plt.axhline(y=harrelldavis_estimate(sample, 0.1))
    plt.show()
