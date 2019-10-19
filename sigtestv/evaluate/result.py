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
