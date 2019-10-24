from collections import defaultdict
import argparse

from tqdm import tqdm
import editdistance as ed
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unlabeled-file', '-i', type=str, required=True)
    parser.add_argument('--labeled-file', '-g', type=str, required=True)
    parser.add_argument('--output-file', '-o', type=str, required=True)
    parser.add_argument('--input-columns', '-c', type=str, nargs='+', default=['sentence'])
    parser.add_argument('--output-column', '-v', type=str, default='label')
    args = parser.parse_args()

    ul_df = pd.read_csv(args.unlabeled_file, sep='\t', quoting=3)
    l_df = pd.read_csv(args.labeled_file, sep='\t', quoting=3)
    is_multicol = len(args.input_columns) > 1
    candidates = {'\t'.join(x) if is_multicol else x: y for (_, x), y in zip(l_df[args.input_columns].itertuples(), l_df[args.output_column])}
    
    output_dict = defaultdict(list)
    for idx, inp_tup in tqdm(list(ul_df[args.input_columns].itertuples())):
        if not is_multicol:
            inp_tup = (inp_tup,)
        best_candidate = max(candidates.items(), key=lambda x: -ed.eval(x[0], '\t'.join(inp_tup)))
        for col, inp in zip(args.input_columns, inp_tup):
            output_dict[col].append(inp)
        del candidates[best_candidate[0]]
        output_dict[args.output_column].append(best_candidate[1])
    pd.DataFrame(output_dict).to_csv(args.output_file, index=False, quoting=3, sep='\t')


if __name__ == '__main__':
    main()
