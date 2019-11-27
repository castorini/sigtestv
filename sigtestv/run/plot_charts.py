from collections import defaultdict
from pathlib import Path
import argparse
import json

from matplotlib import pyplot as plt
from tqdm import trange
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd

from sigtestv.stats import MeanMaxEstimator, BackwardEstimator, ForwardEstimator


def plot_max(ax,
             name,
             results,
             scale_factor=1,
             total=None,
             plot_type='all',
             **estimator_kwargs):
    if total is None:
        total = len(results)
    y = []
    x2 = []
    x3 = []
    pa = []
    p50 = []
    estimator_kwargs['quantile'] = 0.1
    for idx in trange(total):
        estimator_kwargs['n'] = idx + 1
        mme = MeanMaxEstimator(options=estimator_kwargs)
        budget_kwargs = estimator_kwargs.copy()
        fe = ForwardEstimator(options=budget_kwargs)
        options = estimator_kwargs.copy()
        be = BackwardEstimator(options=options)

        y.append(mme.estimate_point(results))
        x2.append(fe.estimate_point(results))
        x3.append(be.estimate_point(results))
    x = scale_factor * (np.arange(total) + 1)
    if plot_type == 'mean':
        ax.plot(x, y, label=f'{name} {mme.name}')
        ax.plot(x2, y, label=f'{name} {fe.name}')
        ax.plot(x3, y, label=f'{name} {be.name}')
        # ax.plot(x, pa, label=name)
    ax.annotate(f'{max(y):.4f}', (max(x) - max(x) // 10, max(y)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-file', '-d', type=Path, required=True)
    parser.add_argument('--column', '-c', type=str, required=True)
    parser.add_argument('--model-name-column', type=str, default='model_name')
    parser.add_argument('--scale-factors', type=json.loads)
    parser.add_argument('--total', '-n', type=int)
    parser.add_argument('--plot-type', '-pt', type=str, default='all', choices=['all', 'p50', 'p10', 'mean'])
    parser.add_argument('--xlabel', '-xl', type=str, default='# Tuning Trials')
    parser.add_argument('--filter-models', '-fm', type=str, nargs='+')
    args = parser.parse_args()

    column_name = args.column
    df = pd.read_csv(args.dataset_file, sep='\t', quoting=3)
    fig, ax = plt.subplots()
    scale_factors = defaultdict(lambda: 1)
    if args.scale_factors is not None:
        scale_factors.update(args.scale_factors)
    for name, group in df.groupby('model_name'):
        if args.filter_models and name not in args.filter_models:
            continue
        results = np.array(list(group[column_name])[:args.total])
        results.sort()
        plot_max(ax, name, results, scale_factor=scale_factors[name], total=args.total, plot_type=args.plot_type)
    plt.legend()
    plt.xlabel(args.xlabel)
    plt.ylabel('Expected Maximum Dev F1')
    plt.title('Model Comparison')
    plt.show()


if __name__ == '__main__':
    main()
