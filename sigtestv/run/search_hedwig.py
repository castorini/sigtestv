import argparse

from sigtestv.evaluate import HedwigExtractor, SearchConfiguration
from .sigseed import run_search_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-iters', '-n', type=int, required=True)
    parser.add_argument('--model-name', type=str, required=True, choices=['han', 'kim_cnn', 'reg_lstm', 'mlp'])
    parser.add_argument('--task-name', type=str, default='Reuters', choices=['Reuters'])
    parser.add_argument('--logger-endpoint', type=str, default='http://0.0.0.0:8080/submit')
    parser.add_argument('--database-file', '-d', type=str, default='bak.db')
    parser.add_argument('--log-file', '-l', type=str, default='log.jsonl')
    parser.add_argument('--format-opt', type=str, default='--workspace')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--base-command', type=str, default='cd /home/ralph/programming/hedwig && '
                                                            'python -u -m models.{model_name}')
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--tabular', action='store_true')
    parser.add_argument('--search-config', '-c', type=str, required=True)
    args = parser.parse_args()
    model_name = args.model_name
    options = {'--dataset': args.task_name, '--epochs': args.epochs}
    args.base_command = args.base_command.format(model_name=model_name)
    run_search_pipeline(args,
                        model_name,
                        options,
                        HedwigExtractor(args.tabular),
                        SearchConfiguration.from_file(args.search_config),
                        capture_stderr=not args.tabular)


if __name__ == '__main__':
    main()
