import numpy as np


def compute_ks_distance(ecdf1: np.ndarray,
                        sample1: np.ndarray,
                        ecdf2: np.ndarray,
                        sample2: np.ndarray):
    max_dist = 0
    max_val = 0
    s1_idx = 0
    s2_idx = 0
    while s1_idx < len(ecdf1) - 1 or s2_idx < len(ecdf2) - 1:
        dist = np.abs(ecdf1[s1_idx] - ecdf2[s2_idx])
        s1 = sample1[s1_idx]
        s2 = sample2[s2_idx]
        if dist > max_dist:
            max_dist = dist
            max_val = s1
        if s1 == s2 and s1_idx != len(ecdf1) - 1 and s2_idx != len(ecdf2) - 1:
            s1_idx += 1
            s2_idx += 1
        elif s1 < s2 and s1_idx != len(ecdf1) - 1:
            s1_idx += 1
        elif s2 < s1 and s2_idx != len(ecdf2) - 1:
            s2_idx += 1
        elif s1_idx == len(ecdf1) - 1:
            s2_idx += 1
        elif s2_idx == len(ecdf2) - 1:
            s1_idx += 1
    return max_val, max_dist
