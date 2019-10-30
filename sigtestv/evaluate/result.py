from dataclasses import dataclass
import enum
from typing import List


class SetTypeEnum(enum.Enum):
    TRAINING = 'training'
    DEV = 'dev'
    TEST = 'test'
    NONE = 'none'


class ResultTypeEnum(enum.Enum):
    ACCURACY = 'accuracy'
    F1 = 'f1'
    SPEARMAN_RHO = 'rho'
    PEARSON_R = 'r'
    MCC = 'mcc'
    LATENCY_MS = 'lat_ms'
    TYPELESS = 'typeless'


class CompletedRun(object):

    def __init__(self, run_config, results, metadata=None):
        self.run_config = run_config
        self.results = results
        self.metadata = metadata


class RunCollection(object):

    def __init__(self, runs: List[CompletedRun]):
        self.runs = runs

    def filter_by_option(self, option_name, option_value):
        runs = list(filter(lambda x: x.run_config.options.get(option_name) == str(option_value), self.runs))
        return RunCollection(runs)

    def filter_by_options(self, option_dict):
        collection = self
        for option_name, option_value in option_dict.items():
            if option_value is not None:
                option_value = str(option_value)
            collection = collection.filter_by_option(option_name, option_value)
        return collection

    def extract_results(self, result_names, set_type):
        result_names = set(result_names)
        rc_results = []
        for run in self.runs:
            results = []
            for result in run.results:
                if result.name in result_names and result.set_type == set_type:
                    results.append(result)
            if len(results) > 0:
                rc_results.append((run.run_config, results))
        return rc_results


@dataclass
class ExperimentResult(object):
    value: float
    name: str = ResultTypeEnum.TYPELESS.value
    set_type: str = SetTypeEnum.NONE.value

    def tolist(self):
        return [self.value, self.name, self.set_type]

    @classmethod
    def fromlist(cls, list_):
        return cls(*list_)
