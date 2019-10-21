import argparse

from sigtestv.evaluate import RunConfiguration, RangeSeedGenerator, EvaluationPipeline, SubprocessRunner, BiRNNExtractor
from sigtestv.net import NetLogger, OfflineNetLogger
from sigtestv.database import ResultsDatabase, DatabaseLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-range', type=int, nargs=2, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--task-name', type=str, default='sst-2', choices=['SST-2'])
    parser.add_argument('--learning-rate', type=float, default=1)
    parser.add_argument('--logger-endpoint', type=str, default='http://0.0.0.0:8080/submit')
    parser.add_argument('--database-file', '-d', type=str, default='bak.db')
    parser.add_argument('--log-file', '-l', type=str, default='log.jsonl')
    parser.add_argument('--base-command', type=str, default='cd /home/ralph/programming/d-bert && '
                                                            'python -m dbert.distill.run.distill_birnn '
                                                            '--config confs/birnn_sst2.json')
    args = parser.parse_args()

    base_command = args.base_command
    dataset_name = args.task_name
    model_name = 'birnn'
    options = {'--lr': str(args.learning_rate),
               '--distill_lambda': 0,
               '--ce_lambda': 1,
               '--batch_size': 50,
               '--mode': 'multichannel',
               '--dropout': 0.1,
               '--workspace': f'{args.output_dir}-{args.task_name}-{model_name}-{{seed}}'}
    config = RunConfiguration(model_name, base_command, dataset_name, options)
    config_generator = RangeSeedGenerator(config, seed_range=args.seed_range, format_opt='--workspace')
    net_logger = NetLogger(args.logger_endpoint)
    db_logger = DatabaseLogger(ResultsDatabase(args.database_file))
    offline_logger = OfflineNetLogger(args.log_file)
    pipeline = EvaluationPipeline(config_generator,
                                  SubprocessRunner(),
                                  [BiRNNExtractor()],
                                  [net_logger, db_logger, offline_logger])
    pipeline()


if __name__ == '__main__':
    main()
