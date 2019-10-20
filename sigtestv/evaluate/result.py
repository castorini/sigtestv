from dataclasses import dataclass
import enum


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


class RunCollection(object):

    def __init__(self, run_configs, results):
        self.run_configs = run_configs
        self.results = results

    def filter_by_option(self, option_name, option_value):
        run_configs, results = list(zip(*list(filter(lambda x: x[0].options.get(option_name) == str(option_value),
                                                     zip(self.run_configs, self.results)))))
        return RunCollection(run_configs, results)

    def filter_by_options(self, option_dict):
        collection = self
        for option_name, option_value in option_dict.items():
            collection = collection.filter_by_option(option_name, option_value)
        return collection

    def extract_results(self, result_name, set_type):
        results = []
        for run_config, results_list in zip(self.run_configs, self.results):
            for result in results_list:
                if result.name == result_name and result.set_type == set_type:
                    results.append((run_config, result.value))
                    break
        return results


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
