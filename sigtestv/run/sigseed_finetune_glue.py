import argparse

from sigtestv.evaluate import BertExtractor
from .sigseed import run_seed_finetuning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--model-name-or-path', type=str, required=True)
    parser.add_argument('--task-name', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--seed-range', type=int, nargs=2)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--seed-iter', type=int)
    parser.add_argument('--learning-rate', type=float, default=4e-5)
    parser.add_argument('--logger-endpoint', type=str, default='http://0.0.0.0:8080/submit')
    parser.add_argument('--database-file', '-d', type=str, default='bak.db')
    parser.add_argument('--log-file', '-l', type=str, default='log.jsonl')
    parser.add_argument('--base-command', type=str, default='python -m sigtestv.run.finetune_glue')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--warmup-steps', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--format-opt', type=str, default='--output_dir')
    args = parser.parse_args()

    model_name = args.model_name_or_path
    options = {'--model_type': args.model_type,
               '--model_name_or_path': args.model_name_or_path,
               '--do_train': None,
               '--do_eval': None,
               '--do_lower_case': None,
               '--task_name': args.task_name,
               '--data_dir': f'{args.data_dir}/{args.task_name}',
               '--max_seq_length': 128,
               '--train_batch_size': args.batch_size,
               '--learning_rate': str(args.learning_rate),
               '--num_train_epochs': 3.0,
               '--save_steps': 10000,
               '--overwrite_output_dir': None,
               '--no_save_checkpoint': None,
               '--no_reload_after_save': None,
               '--output_dir': f'{args.output_dir}-{args.task_name}-{args.model_name_or_path}-{{seed}}'}
    if args.warmup_steps:
        options['--warmup_steps'] = args.warmup_steps
    if args.max_steps:
        options['--max_steps'] = args.max_steps
    run_seed_finetuning(args, model_name, options, BertExtractor())


if __name__ == '__main__':
    main()
