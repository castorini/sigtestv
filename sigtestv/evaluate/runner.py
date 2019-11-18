from dataclasses import dataclass
from typing import Dict
import enum
import io
import subprocess
import os

from .pipeline import PipelineComponent, EvaluationPipeline


class PassTypeEnum(enum.Enum):
    CLI = 'cli'
    ENV = 'env'


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
        self.options = {k: v if v is None else str(v) for k, v in self.options.items()}
        self.env_vars = {k: v if v is None else str(v) for k, v in self.env_vars.items()}

    @property
    def hyperparameters(self):
        hyperparams = self.options.copy()
        hyperparams.update(self.env_vars)
        return hyperparams

    def attr(self, name):
        return self.hyperparameters.get(name)

    def check_requires(self, bool_expr):
        return bool_expr.evaluate(set(self.hyperparameters.keys()))


class SubprocessRunner(PipelineComponent):

    def __init__(self, line_write_callback=None, capture_stderr=False):
        self.line_write_cb = line_write_callback
        self.capture_stderr = capture_stderr

    def __call__(self, run_config):
        command_args = [run_config.command_base]
        sio = io.StringIO()
        for k, v in run_config.options.items():
            command_args.append(k)
            if v: command_args.append(v)
        stderr_opt = dict(stderr=subprocess.STDOUT) if self.capture_stderr else {}
        result = subprocess.Popen(' '.join(command_args),
                                  shell=True,
                                  env=run_config.env_vars,
                                  stdout=subprocess.PIPE,
                                  bufsize=0,
                                  **stderr_opt)
        for line in iter(result.stdout.readline, b''):
            line = line.decode()
            sio.write(line)
            if self.line_write_cb is not None:
                self.line_write_cb(line.rstrip())
        return sio.getvalue()


class ConfigGenerator(object):

    def __iter__(self):
        raise NotImplementedError

    def load_state_dict(self, sd):
        pass

    def state_dict(self):
        return {}

    def __len__(self):
        raise NotImplementedError


class SeedConfigGenerator(ConfigGenerator):

    def __init__(self,
                 base_config,
                 pass_type='cli',
                 seed_range=(0, 100),
                 env_opt_name='SEED',
                 cli_opt_name='--seed',
                 format_opt='--output_dir',
                 seeds=None):
        if seeds is None:
            seeds = list(range(*seed_range))
        self.seeds = seeds
        self.pass_type = pass_type
        self.base_config = base_config
        self.env_opt_name = env_opt_name
        self.cli_opt_name = cli_opt_name
        self.format_opt = format_opt

    def __iter__(self):
        env = os.environ.copy()
        format_str = self.base_config.options[self.format_opt]
        for seed in self.seeds:
            if self.pass_type == 'cli':
                self.base_config.options[self.cli_opt_name] = str(seed)
            else:
                env[self.env_opt_name] = str(seed)
                self.base_config.env_vars = env
            if self.format_opt in self.base_config.options:
                self.base_config.options[self.format_opt] = format_str.format(seed=seed)
            yield self.base_config, dict(seed=seed)

    def __len__(self):
        return len(self.seeds)


class IdentityConfigWrapper(ConfigGenerator):

    def __init__(self, base_config, metadata=None):
        self.config = base_config
        self.metadata = metadata

    def __iter__(self):
        yield self.config, self.metadata

    def __len__(self):
        return 1


if __name__ == '__main__':
    from .extract import PingExtractor
    from sigtestv.net import NetLogger
    from sigtestv.database import DatabaseLogger, ResultsDatabase
    print(SubprocessRunner()(RunConfiguration('list', 'ls', '', {'-laht': None})))
    print(((bexpr('a') & bexpr('b')) | bexpr('c')).evaluate({'c'}))
    config_generator = SeedConfigGenerator(RunConfiguration('ping_test', 'ping asdf.com', ''),
                                           seed_range=(1, 100),
                                           cli_opt_name='-c')
    # logger = DatabaseLogger(ResultsDatabase('tmptest.db'))
    logger = NetLogger('http://dragon.cs.uwaterloo.ca:8080/submit')
    pipeline = EvaluationPipeline(config_generator, SubprocessRunner(), [PingExtractor()], [logger])
    pipeline()
