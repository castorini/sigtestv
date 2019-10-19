from dataclasses import dataclass
from typing import Dict
import subprocess
import os

from .pipeline import PipelineComponent, EvaluationPipeline


class BoolSetExpression(object):

    def __init__(self, elements, expr_type='none'):
        self.elements = elements
        self.expr_type = expr_type

    def add_expr(self, expr, expr_type):
        if expr.expr_type == 'none' or expr.expr_type != self.expr_type:
            return BoolSetExpression([self, expr], expr_type=expr_type)
        else:
            self.elements.extend(expr.elements)
            return self

    def __or__(self, expr):
        return self.add_expr(expr, 'or')

    def __and__(self, expr):
        return self.add_expr(expr, 'and')

    def evaluate(self, set_):
        if self.expr_type == 'and':
            return all(e.evaluate(set_) for e in self.elements)
        elif self.expr_type == 'or':
            return any(e.evaluate(set_) for e in self.elements)
        else:
            return len(self.elements.intersection(set_)) == len(self.elements)


def bexpr(*args):
    return BoolSetExpression(set(args))


@dataclass
class RunConfiguration(object):
    model_name: str
    command_base: str
    dataset_name: str
    options: Dict[str, str] = None
    env_vars: Dict[str, str] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}
        if self.env_vars is None:
            self.env_vars = os.environ

    @property
    def hyperparameters(self):
        hyperparams = self.options.copy()
        hyperparams.update(self.env_vars)
        return hyperparams

    def check_requires(self, bool_expr):
        return bool_expr.evaluate(set(self.hyperparameters.keys()))


class SubprocessRunner(PipelineComponent):

    def __call__(self, run_config):
        command_args = [run_config.command_base]
        for k, v in run_config.options.items():
            command_args.append(k)
            if v: command_args.append(v)
        result = subprocess.run(' '.join(command_args), shell=True, capture_output=True, env=run_config.env_vars)
        print(result.stderr.decode())
        return result.stdout.decode()


class ConfigGenerator(object):

    def __iter__(self):
        raise NotImplementedError

    def load_state_dict(self, sd):
        pass

    def state_dict(self):
        return {}

    def __len__(self):
        raise NotImplementedError


class RangeSeedGenerator(ConfigGenerator):

    def __init__(self,
                 base_config,
                 pass_type='cli',
                 seed_range=(0, 100),
                 env_opt_name='SEED',
                 cli_opt_name='--seed',
                 format_opt='--output_dir'):
        self.seed_range = seed_range
        self.pass_type = pass_type
        self.base_config = base_config
        self.env_opt_name = env_opt_name
        self.cli_opt_name = cli_opt_name
        self.format_opt = format_opt

    def __iter__(self):
        env = os.environ.copy()
        for seed in range(*self.seed_range):
            if self.pass_type == 'cli':
                self.base_config.options[self.cli_opt_name] = str(seed)
            else:
                env[self.env_opt_name] = str(seed)
                self.base_config.env_vars = env
            if self.format_opt in self.base_config.options:
                self.base_config.options[self.format_opt] = self.base_config.options[self.format_opt].format(seed=seed)
            yield self.base_config

    def __len__(self):
        return self.seed_range[1] - self.seed_range[0]


if __name__ == '__main__':
    from .extract import PingExtractor
    from sigtestv.net import NetLogger
    from sigtestv.database import DatabaseLogger, ResultsDatabase
    print(SubprocessRunner()(RunConfiguration('list', 'ls', '', {'-laht': None})))
    print(((bexpr('a') & bexpr('b')) | bexpr('c')).evaluate({'c'}))
    config_generator = RangeSeedGenerator(RunConfiguration('ping_test', 'ping asdf.com', ''),
                                          seed_range=(1, 100),
                                          cli_opt_name='-c')
    # logger = DatabaseLogger(ResultsDatabase('tmptest.db'))
    logger = NetLogger('http://0.0.0.0:5358/submit')
    pipeline = EvaluationPipeline(config_generator, SubprocessRunner(), [PingExtractor()], [logger])
    pipeline()
