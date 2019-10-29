from pathlib import Path
import argparse

from sigtestv.evaluate import JiantExtractor
from .sigseed import run_seed_finetuning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-prefix', '-pp', type=str, required=True)
    parser.add_argument('--data-dir', '-d', type=Path, required=True)
    parser.add_argument('--word-embeddings-file', '-emb', type=Path, required=True)
    parser.add_argument('--config-file', '-c', type=Path, default='confs/jiant_base.conf')
    parser.add_argument('--seed-iter', type=int, required=True)
    parser.add_argument('--task-name', type=str, default='sts-b', choices=['sts-b'])
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--logger-endpoint', type=str, default='http://0.0.0.0:8080/submit')
    parser.add_argument('--database-file', '-db', type=str, default='bak.db')
    parser.add_argument('--log-file', '-l', type=str, default='jiant-log.jsonl')
    parser.add_argument('--format-opt', type=str, default='--run-name')
    parser.add_argument('--base-command', type=str, default='python -m sigtestv.run.finetune_jiant_glue')
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    model_name = 'birnn-jiant'
    options = {'--lr': args.learning_rate,
               '--run-name': f'{model_name}-{{seed}}',
               '--task-name': args.task_name,
               '-emb': args.word_embeddings_file,
               '-c': args.config_file,
               '-pp': args.project_prefix,
               '-d': args.data_dir}
    run_seed_finetuning(args, model_name, options, JiantExtractor())


if __name__ == '__main__':
    main()
