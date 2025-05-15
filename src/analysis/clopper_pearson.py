# statistics/clopper_pearson.py

"""
Statistical routines for determinism tests.
"""

from scipy.stats import beta


def clopper_pearson(s: int, N: int, alpha: float):
    """
    Compute Clopper–Pearson confidence interval for s successes in N trials.
    Handles edge cases to avoid NaNs.
    Returns (pmin, pmax).
    """
    if s == 0:
        pmin = 0.0
        pmax = beta.ppf(1 - alpha / 2, 1, N)
    elif s == N:
        pmin = beta.ppf(alpha / 2, N, 1)
        pmax = 1.0
    else:
        pmin = beta.ppf(alpha / 2, s, N - s + 1)
        pmax = beta.ppf(1 - alpha / 2, s + 1, N - s)
    return pmin, pmax