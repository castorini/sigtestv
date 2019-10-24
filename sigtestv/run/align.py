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

    l_df = pd.read_csv(args.labeled_file, sep='\t', quoting=3)
    ul_df = pd.read_csv(args.unlabeled_file, sep='\t', quoting=3)
    is_multicol = len(args.input_columns) > 1
    candidates = [('\t'.join(x[1:]), y) for x, y in zip(l_df[args.input_columns].itertuples(), l_df[args.output_column])]
    
    output_dict = defaultdict(list)
    columns = list(ul_df.columns) + [args.output_column]
    for tup in tqdm(list(ul_df[list(ul_df.columns)].itertuples())):
        tup = tup[1:]
        for col, x in zip(ul_df.columns, tup):
            output_dict[col].append(x)
    for inp_tup in tqdm(list(ul_df[args.input_columns].itertuples())):
        inp_tup = inp_tup[1:]
        best_candidate = min(candidates, key=lambda x: ed.eval(x[0], '\t'.join(inp_tup)))
        candidates.pop(candidates.index(best_candidate))
        output_dict[args.output_column].append(best_candidate[1])
    pd.DataFrame(output_dict)[columns].to_csv(args.output_file, index=False, quoting=3, sep='\t')


if __name__ == '__main__':
    main()
