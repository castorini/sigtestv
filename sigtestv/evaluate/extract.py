import pathlib
import re

from .runner import PipelineComponent, RunConfiguration, bexpr
from .result import ExperimentResult, SetTypeEnum


class BertExtractor(PipelineComponent):

    def __call__(self, config: RunConfiguration, stdout: str):
        output_dir = config.hyperparameters.get('--output_dir')
        if output_dir is None: output_dir = config.hyperparameters['-o']
        path = pathlib.Path(output_dir) / 'eval_results.txt'
        with open(path) as f:
            lines = f.readlines()
        results = []
        for line in lines:
            line = line.strip()
            name, value = line.split(' = ')
            results.append(ExperimentResult(float(value), name, SetTypeEnum.DEV.value))
        return results

    def requires(self):
        return bexpr('--output_dir') | bexpr('-o')


class PingExtractor(PipelineComponent):

    def __call__(self, config, stdout):
        values = re.match(r'^.*\s(.+?)/(.+?)/(.+?)/(.+?)\s.*$', stdout.strip().split('\n')[-1]).groups()
        min_, avg, max_, mdev = list(map(float, values))
        return list(map(lambda x: ExperimentResult(*x), ((min_, 'lat min'), (avg, 'lat avg'), (max_, 'lat max'), (mdev, 'lat mdev'))))
