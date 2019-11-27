import numpy as np


def cv_adjust(samples, control_samples, expected_control, cov=None):
    if cov is None:
        cov = np.cov(samples, control_samples)[0][1]
    var = np.var(control_samples)
    if var == 0: return samples
    c = -cov / var
    return samples + c * (control_samples - expected_control)

