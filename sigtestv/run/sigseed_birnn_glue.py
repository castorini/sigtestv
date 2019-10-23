import argparse

from .sigseed import run_seed_finetuning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-range', type=int, nargs=2)
    parser.add_argument('--seed-iter', type=int)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--task-name', type=str, default='sst-2', choices=['SST-2'])
    parser.add_argument('--learning-rate', type=float, default=1)
    parser.add_argument('--logger-endpoint', type=str, default='http://0.0.0.0:8080/submit')
    parser.add_argument('--database-file', '-d', type=str, default='bak.db')
    parser.add_argument('--log-file', '-l', type=str, default='log.jsonl')
    parser.add_argument('--format-opt', type=str, default='--workspace')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--base-command', type=str, default='cd /home/ralph/programming/d-bert && '
                                                            'python -m dbert.distill.run.distill_birnn '
                                                            '--config confs/birnn_sst2.json')
    parser.add_argument('--online', action='store_true')
    args = parser.parse_args()
    model_name = 'birnn'
    options = {'--lr': str(args.learning_rate),
               '--distill_lambda': 0,
               '--ce_lambda': 1,
               '--epochs': args.epochs,
               '--batch_size': 50,
               '--mode': 'multichannel',
               '--dropout': 0.1,
               '--workspace': f'{args.output_dir}-{args.task_name}-{model_name}-{{seed}}'}
    run_seed_finetuning(args, model_name, options)


if __name__ == '__main__':
    main()
