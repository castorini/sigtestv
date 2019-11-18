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


class HedwigExtractor(PipelineComponent):

    def __call__(self, config: RunConfiguration, stdout: str):
        def extract_results(line1, line2, line3):
            set_type = line1.split('for ')[-1]
            metric_names = eval(line2)
            results = eval(line3)
            return [ExperimentResult(float(value), name, set_type) for value, name in zip(results, metric_names)]
        lines = stdout.splitlines()[-6:]
        dev_lines = lines[:3]
        test_lines = lines[3:]
        return extract_results(*dev_lines) + extract_results(*test_lines)


class JiantExtractor(PipelineComponent):

    def __init__(self, set_type=SetTypeEnum.DEV):
        self.set_type = set_type.value

    def __call__(self, config: RunConfiguration, stdout: str):
        lines = stdout.splitlines()
        idx = 0
        for idx, line in enumerate(lines):
            if 'VALIDATION RESULTS' in line:
                break
        return [ExperimentResult(float(v), k, self.set_type) for k, v in re.findall(r'_(\w+?): (\d+\.\d+)', lines[idx + 1])]


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
