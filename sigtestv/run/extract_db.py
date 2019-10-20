from collections import defaultdict
import argparse
import sys

import pandas as pd

from sigtestv.database import ResultsDatabase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database-file', '-f', type=str, required=True)
    parser.add_argument('--model-name', '-m', type=str, required=True)
    parser.add_argument('--dataset-name', '-d', type=str, required=True)
    parser.add_argument('--result-name', '-n', type=str, required=True)
    parser.add_argument('--set-type', '-s', type=str, default='dev')
    parser.add_argument('--option-names', '-o', type=str, nargs='+', default=[])
    parser.add_argument('--option-values', '-v', type=str, nargs='+', default=[])
    parser.add_argument('--extract-options', '-x', type=str, nargs='+', default=[])
    args = parser.parse_args()
    args.option_names = list(map(str.strip, args.option_names))
    args.extract_options = list(map(str.strip, args.extract_options))

    database = ResultsDatabase(args.database_file)
    run_collection = database.fetch_all(args.model_name, args.dataset_name)
    run_collection = run_collection.filter_by_options(dict(zip(args.option_names, args.option_values)))
    results = run_collection.extract_results(args.result_name, args.set_type)

    df_data = defaultdict(list)
    for rc, result in results:
        df_data['model_name'].append(rc.model_name)
        df_data['dataset_name'].append(rc.dataset_name)
        for opt_name in args.extract_options:
            df_data[opt_name].append(rc.attr(opt_name))
        df_data['set_type'].append(args.set_type)
        df_data['result_name'].append(args.result_name)
        df_data['result'].append(result)

    pd.DataFrame(df_data).to_csv(sys.stdout, sep='\t', quoting=3, index=False)


if __name__ == '__main__':
    main()