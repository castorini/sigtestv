import secrets

from tqdm import tqdm

from sigtestv.evaluate import RunConfiguration, SeedConfigGenerator, EvaluationPipeline, SubprocessRunner, IdentityConfigWrapper,\
    SearchConfiguration, RandomSearchGenerator
from sigtestv.net import NetLogger, OfflineNetLogger
from sigtestv.database import ResultsDatabase, DatabaseLogger, DatabaseUpdateLogger


def run_search_pipeline(args,
                        model_name,
                        options,
                        extractor,
                        search_config: SearchConfiguration,
                        **kwargs):
    base_command = args.base_command
    dataset_name = args.task_name
    total = args.num_iters
    config = RunConfiguration(model_name, base_command, dataset_name, options)
    config_generator = RandomSearchGenerator(config, search_config, total, format_opt=args.format_opt)
    net_logger = NetLogger(args.logger_endpoint)
    db_logger = DatabaseLogger(ResultsDatabase(args.database_file))
    offline_logger = OfflineNetLogger(args.log_file)
    loggers = [db_logger, offline_logger]
    if args.online: loggers.append(net_logger)
    pipeline = EvaluationPipeline(config_generator,
                                  SubprocessRunner(line_write_callback=tqdm.write, **kwargs),
                                  [extractor],
                                  loggers)
    pipeline()


def run_seed_finetuning(args, model_name, options, extractor):
    if not hasattr(args, 'seed_range') or args.seed_range is None:
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
                                  SubprocessRunner(line_write_callback=tqdm.write),
                                  [extractor],
                                  loggers)
    pipeline()


def run_pipeline(args, model_name, options, extractor, metadata=None):
    base_command = args.base_command
    dataset_name = args.task_name
    config = RunConfiguration(model_name, base_command, dataset_name, options)
    config_generator = IdentityConfigWrapper(config, metadata=metadata)
    net_logger = NetLogger(args.logger_endpoint)
    db_logger = DatabaseUpdateLogger(ResultsDatabase(args.database_file))
    offline_logger = OfflineNetLogger(args.log_file)
    loggers = [db_logger, offline_logger]
    if args.online: loggers.append(net_logger)
    pipeline = EvaluationPipeline(config_generator,
                                  SubprocessRunner(),
                                  [extractor],
                                  loggers)
    pipeline(use_tqdm=False)