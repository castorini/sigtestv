from pathlib import Path
import argparse

from tqdm import tqdm
import pandas as pd

from sigtestv.stats import ResultPopulationPair, ResultPopulationSingle, StudentsTTest, ASDTest, MannWhitneyUTest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', '--file', '-f', type=Path, nargs='+', required=True)
    parser.add_argument('--num-samples', '-n', type=int, required=True)
    parser.add_argument('--result-column-name', '-c', type=str, required=True)
    parser.add_argument('--num-samples2', '-n2', type=int)
    parser.add_argument('--result-column-name2', '-c2', type=str)
    parser.add_argument('--num-iters', '-it', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--asd-threshold', '--asd-th', type=float, default=0.5)
    parser.add_argument('--test', type=str, default='power', choices=['power', 'type1'])
    args = parser.parse_args()

    if args.result_column_name2 is None:
        args.result_column_name2 = args.result_column_name
    if args.num_samples2 is None:
        args.num_samples2 = args.num_samples

    ttest = StudentsTTest(dict(equal_var=False))
    asd_test = ASDTest(dict(threshold=args.asd_threshold))
    mwu_test = MannWhitneyUTest()
    tests = (ttest, asd_test, mwu_test)

    pop_x = pd.read_csv(args.files[0], sep='\t', quoting=3)[args.result_column_name]
    if args.test == 'power':
        pop_y = pd.read_csv(args.files[1], sep='\t', quoting=3)[args.result_column_name2]
        pop_pair = ResultPopulationPair(pop_x, pop_y)
        results = pop_pair.simulate_statistical_power(args.num_samples,
                                                      tests,
                                                      use_tqdm=True,
                                                      n2=args.num_samples2,
                                                      iters=args.num_iters,
                                                      alpha=args.alpha)
    else:
        pop_single = ResultPopulationSingle(pop_x)
        results = pop_single.simulate_type1_error(args.num_samples,
                                                  tests,
                                                  use_tqdm=True,
                                                  iters=args.num_iters,
                                                  alpha=args.alpha)

    print('name\tresult')
    for test_name, result in results.items():
        print(f'{test_name}\t{result:.4f}')


if __name__ == '__main__':
    main()
