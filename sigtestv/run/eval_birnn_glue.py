import argparse

from tqdm import tqdm

from .sigseed import run_pipeline
from sigtestv.database import ResultsDatabase
from sigtestv.evaluate import BiRNNCliExtractor, SetTypeEnum
from sigtestv.utils.list import chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-dd', type=str, required=True)
    parser.add_argument('--task-name', type=str, default='SST-2', choices=['SST-2'])
    parser.add_argument('--learning-rate', type=float, default=1)
    parser.add_argument('--logger-endpoint', type=str, default='http://0.0.0.0:8080/submit')
    parser.add_argument('--database-file', '-d', type=str, default='bak.db')
    parser.add_argument('--log-file', '-l', type=str, default='log.jsonl')
    parser.add_argument('--format-opt', type=str, default='--workspace')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--base-command', type=str, default='cd /home/ralph/programming/d-bert && '
                                                            'python -m dbert.distill.run.distill_birnn '
                                                            '--config confs/birnn_sst2.json')
    parser.add_argument('--chunk-idx', type=int, default=0)
    parser.add_argument('--chunks', type=int, default=1)
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()

    db = ResultsDatabase(args.database_file)
    model_name = 'birnn'
    rc = db.fetch_all(model_name, args.task_name)
    rc = rc.filter_by_options({'--lr': args.learning_rate})
    for run in tqdm(list(chunk(rc.runs, args.chunks))[args.chunk_idx]):
        options = run.run_config.options
        exp_id = run.metadata['exp_id']
        options['--dataset_path'] = args.data_dir
        options['--load_best_checkpoint'] = None
        options['--eval_test_labeled'] = None
        run_pipeline(args, model_name, options, BiRNNCliExtractor(set_type=SetTypeEnum.TEST), metadata=dict(exp_id=exp_id))


if __name__ == '__main__':
    main()
