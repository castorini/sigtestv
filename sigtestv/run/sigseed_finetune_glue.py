import argparse

from sigtestv.evaluate import RunConfiguration, RangeSeedGenerator, EvaluationPipeline, SubprocessRunner, BertExtractor
from sigtestv.net import NetLogger, OfflineNetLogger
from sigtestv.database import ResultsDatabase, DatabaseLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True)
    parser.add_argument('--model-name-or-path', type=str, required=True)
    parser.add_argument('--task-name', type=str, required=True)
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--seed-range', type=int, nargs=2, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--learning-rate', type=float, default=4e-5)
    parser.add_argument('--logger-endpoint', type=str, default='http://0.0.0.0:8080/submit')
    parser.add_argument('--database-file', '-d', type=str, default='bak.db')
    parser.add_argument('--log-file', '-l', type=str, default='log.jsonl')
    args = parser.parse_args()

    base_command = 'python -m sigtestv.run.finetune_glue'
    dataset_name = args.task_name
    model_name = args.model_name_or_path
    options = {'--model_type': args.model_type,
               '--model_name_or_path': args.model_name_or_path,
               '--do_train': None,
               '--do_eval': None,
               '--do_lower_case': None,
               '--task_name': args.task_name,
               '--data_dir': f'{args.data_dir}/{args.task_name}',
               '--max_seq_length': '128',
               '--train_batch_size': '32',
               '--learning_rate': str(args.learning_rate),
               '--num_train_epochs': '3.0',
               '--save_steps': '10000',
               '--overwrite_output_dir': None,
               '--output_dir': f'{args.output_dir}-{args.task_name}-{args.model_name_or_path}-{{seed}}'}
    config = RunConfiguration(model_name, base_command, dataset_name, options)
    config_generator = RangeSeedGenerator(config, seed_range=args.seed_range)
    net_logger = NetLogger(args.logger_endpoint)
    db_logger = DatabaseLogger(ResultsDatabase(args.database_file))
    offline_logger = OfflineNetLogger(args.log_file)
    pipeline = EvaluationPipeline(config_generator,
                                  SubprocessRunner(),
                                  [BertExtractor()],
                                  [net_logger, db_logger, offline_logger])
    pipeline()


if __name__ == '__main__':
    main()
