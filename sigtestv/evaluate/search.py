from dataclasses import dataclass
from functools import partial
from typing import Sequence, Any
import json
import os

import numpy as np
import scipy.stats as stats

from .runner import ConfigGenerator, PassTypeEnum


@dataclass
class AttributeSample(object):
    name: str
    value: Any
    pass_type: str


@dataclass
class SearchConfiguration(object):
    names: Sequence[str]
    sample_functions: Sequence[partial]
    pass_types: Sequence[str]

    @classmethod
    def from_list(cls, data_lst):
        names = []
        sample_functions = []
        pass_types = []
        for attr_dict in data_lst:
            names.append(attr_dict['name'])
            del attr_dict['name']
            try:
                pass_types.append(attr_dict['pass_type'])
                del attr_dict['pass_type']
            except KeyError:
                pass_types.append(PassTypeEnum.CLI.value)

            try:
                sampling_fn = attr_dict['sampling_fn']
                del attr_dict['sampling_fn']
                sampling_fn = getattr(stats, sampling_fn).rvs
                sampling_fn = partial(sampling_fn, **attr_dict)
            except (KeyError, AttributeError):
                choices = attr_dict['choices']
                del attr_dict['choices']
                sampling_fn = partial(np.random.choice, choices, **attr_dict)
            sample_functions.append(sampling_fn)
        return cls(names, sample_functions, pass_types)

    @classmethod
    def from_file(cls, filename: str):
        with open(filename) as f:
            data = json.load(f)
        return cls.from_list(data)

    def sample(self):
        samples = (fn() for fn in self.sample_functions)
        return [AttributeSample(*x) for x in zip(self.names, samples, self.pass_types)]


class RandomSearchGenerator(ConfigGenerator):

    def __init__(self,
                 base_config,
                 search_config: SearchConfiguration,
                 total: int,
                 format_opt='--output_dir'):
        self.base_config = base_config
        self.search_config = search_config
        self.format_opt = format_opt
        self.total = total

    def __iter__(self):
        for _ in range(self.total):
            env = os.environ.copy()
            format_str = self.base_config.options.get(self.format_opt, '')
            attr_samples = self.search_config.sample()
            attr_dict = {attr_sample.name: attr_sample.value for attr_sample in attr_samples}

            for attr_sample in attr_samples:
                pass_type = attr_sample.pass_type
                if pass_type == PassTypeEnum.CLI.value:
                    self.base_config.options[attr_sample.name] = str(attr_sample.value)
                elif pass_type == PassTypeEnum.ENV.value:
                    env[attr_sample.name] = str(attr_sample.value)
                    self.base_config.env_vars = env
                else:
                    raise ValueError(f'Unknown pass type {pass_type}')
            if self.format_opt in self.base_config.options:
                self.base_config.options[self.format_opt] = format_str.format(**attr_dict)
            yield self.base_config, attr_dict

    def __len__(self):
        return self.total
