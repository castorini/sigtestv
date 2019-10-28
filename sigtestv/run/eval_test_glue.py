import argparse

from tqdm import tqdm

from sigtestv.database import ResultsDatabase
from sigtestv.evaluate import BertExtractor, SetTypeEnum
from .sigseed import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--model-name-or-path', type=str, required=True)
    parser.add_argument('--task-name', type=str, required=True)
    parser.add_argument('--data-dir', '-dd', type=str, required=True)
    parser.add_argument('--database-file', '-df', type=str, required=True)
    parser.add_argument('--seed-iter', type=int)
    parser.add_argument('--learning-rate', type=float, default=4e-5)
    parser.add_argument('--logger-endpoint', type=str, default='http://0.0.0.0:8080/submit')
    parser.add_argument('--log-file', '-l', type=str, default='log.jsonl')
    parser.add_argument('--base-command', type=str, default='python -m sigtestv.run.finetune_glue')
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()

    db = ResultsDatabase(args.database_file)
    model_name = args.model_name_or_path
    rc = db.fetch_all(model_name, args.task_name)
    rc = rc.filter_by_options({'--learning_rate': args.learning_rate,
                               '--model_type': args.model_type})
    for config in tqdm(rc.run_configs):
        options = config.options
        del options['--do_train']
        options['--data-dir'] = f'{args.data_dir}/{args.task_name}'
        run_pipeline(args, model_name, options, BertExtractor(set_type=SetTypeEnum.TEST))


if __name__ == '__main__':
    main()
