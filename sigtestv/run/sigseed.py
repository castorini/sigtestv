import secrets

from sigtestv.evaluate import RunConfiguration, SeedConfigGenerator, EvaluationPipeline, SubprocessRunner, BiRNNExtractor
from sigtestv.net import NetLogger, OfflineNetLogger
from sigtestv.database import ResultsDatabase, DatabaseLogger


def run_seed_finetuning(args, model_name, options):
    if args.seed_range is None:
        seeds = [secrets.randbits(31) for _ in range(args.seed_iter)]
    else:
        seeds = list(range(*args.seed_range))

    base_command = args.base_command
    dataset_name = args.task_name
    config = RunConfiguration(model_name, base_command, dataset_name, options)
    config_generator = SeedConfigGenerator(config, seeds=seeds, format_opt=args.format_opt)
    net_logger = NetLogger(args.logger_endpoint)
    db_logger = DatabaseLogger(ResultsDatabase(args.database_file))
    offline_logger = OfflineNetLogger(args.log_file)
    loggers = [db_logger, offline_logger]
    if args.online: loggers.append(net_logger)
    pipeline = EvaluationPipeline(config_generator,
                                  SubprocessRunner(),
                                  [BiRNNExtractor()],
                                  loggers)
    pipeline()
