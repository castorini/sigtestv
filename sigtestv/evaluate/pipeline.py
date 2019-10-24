from dataclasses import dataclass
from tqdm import tqdm
import abc

from .result import CompletedRun


class PipelineComponent(object):

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    def requires(self):
        from sigtestv.evaluate import bexpr
        return bexpr()


class EvaluationPipeline(object):

    def __init__(self, config_generator, runner, extractors, loggers):
        self.config_generator = config_generator
        self.runner = runner
        self.extractors = extractors
        self.loggers = loggers
        self.state_dict = self.config_generator.state_dict()
        self.generator_idx = 0

    def load_state_dict(self, state_dict):
        self.config_generator.load_state_dict(state_dict['config_generator'])
        self.generator_idx = state_dict['generator_idx']

    def state_dict(self):
        return dict(generator_idx=self.generator_idx, config_generator=self.config_generator.state_dict())

    def __call__(self, use_tqdm=True):
        gen_iter = iter(self.config_generator)
        for _ in range(self.generator_idx): next(gen_iter)
        for config, metadata in tqdm(gen_iter,
                           total=len(self.config_generator),
                           initial=self.generator_idx,
                           desc='Running pipeline',
                           disable=not use_tqdm):
            config.check_requires(self.runner.requires())
            for ex in self.extractors:
                config.check_requires(ex.requires())
            for logger in self.loggers:
                config.check_requires(logger.requires())
            stdout = self.runner(config)
            exp_results = []
            for ex in self.extractors:
                ret_val = ex(config, stdout)
                if isinstance(ret_val, list):
                    exp_results.extend(ret_val)
                else:
                    exp_results.append(ret_val)
            for logger in self.loggers:
                logger(CompletedRun(config, exp_results, metadata=metadata))
            self.generator_idx += 1


if __name__ == '__main__':
    pipeline = EvaluationPipeline()