from dataclasses import dataclass
from typing import Dict
import subprocess
import os


@dataclass
class RunConfiguration(object):
    model_name: str
    command_base: str
    options: Dict[str, str] = None

    def __post_init__(self):
        if self.options is None:
            self.options = {}
        self.env_vars = os.environ


@dataclass
class SubprocessRunner(object):
    run_config: RunConfiguration

    def __call__(self):
        command_args = [self.run_config.command_base]
        for k, v in self.run_config.options.items():
            command_args.append(k)
            if v: command_args.append(v)
        return subprocess.run(command_args, capture_output=True).stdout.decode()


if __name__ == '__main__':
    print(SubprocessRunner(RunConfiguration('list', 'ls', {'-laht': None}))())