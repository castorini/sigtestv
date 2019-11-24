import numpy as np


def cv_adjust(samples, control_samples, expected_control):
    cov = np.cov(samples, control_samples)[0][1]
    c = -cov / np.var(control_samples)
    return samples + c * (control_samples - expected_control)
