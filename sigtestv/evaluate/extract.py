import pathlib
import re

import torch

from .runner import PipelineComponent, RunConfiguration, bexpr
from .result import ExperimentResult, SetTypeEnum


class BertExtractor(PipelineComponent):

    def __init__(self, output_dir_key='--output_dir', set_type=SetTypeEnum.DEV):
        self.output_dir_key = output_dir_key
        self.set_type = set_type.value

    def __call__(self, config: RunConfiguration, stdout: str):
        output_dir = config.hyperparameters.get(self.output_dir_key)
        if output_dir is None: output_dir = config.hyperparameters['-o']
        path = pathlib.Path(output_dir) / 'eval_results.txt'
        with open(path) as f:
            lines = f.readlines()
        results = []
        for line in lines:
            line = line.strip()
            name, value = line.split(' = ')
            results.append(ExperimentResult(float(value), name, self.set_type))
        return results

    def requires(self):
        return bexpr(self.output_dir_key)


class BiRNNExtractor(PipelineComponent):

    def __init__(self, output_dir_key='--workspace', set_type=SetTypeEnum.DEV):
        self.output_dir_key = output_dir_key
        self.set_type = set_type.value

    def __call__(self, config: RunConfiguration, stdout: str):
        output_dir = config.hyperparameters.get(self.output_dir_key)
        path = pathlib.Path(output_dir) / 'best_model.pt'
        sd = torch.load(path)
        results = []
        for key, value in sd.items():
            if not key.startswith('dev_'):
                continue
            name = key[4:]
            results.append(ExperimentResult(value, name, self.set_type))
        return results

    def requires(self):
        return bexpr(self.output_dir_key)


class BiRNNCliExtractor(PipelineComponent):

    def __init__(self, set_type=SetTypeEnum.DEV):
        self.set_type = set_type.value

    def __call__(self, config: RunConfiguration, stdout: str):
        results = re.findall('(.+?) = (.+?)\n', stdout)
        return [ExperimentResult(float(value), name, self.set_type) for name, value in results]


class PingExtractor(PipelineComponent):

    def __call__(self, config, stdout):
        values = re.match(r'^.*\s(.+?)/(.+?)/(.+?)/(.+?)\s.*$', stdout.strip().split('\n')[-1]).groups()
        min_, avg, max_, mdev = list(map(float, values))
        return list(map(lambda x: ExperimentResult(*x), ((min_, 'lat min'), (avg, 'lat avg'), (max_, 'lat max'), (mdev, 'lat mdev'))))
