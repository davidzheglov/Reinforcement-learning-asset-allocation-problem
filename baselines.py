"""
baselines.py -- Hold and heuristic baseline policies.
"""

import numpy as np


def hold_policy(obs, env):
    """Do nothing: zero action (keep current allocation)."""
    return np.zeros(env.n, dtype=np.float32)


def heuristic_policy(obs, env):
    """
    Sharpe-score heuristic: allocate proportional to excess return / std.
    Maps to [-1, 1]^n via normalisation.
    """
    excess = env.a_k - env.r
    std = np.sqrt(env.s_k)
    scores = excess / np.maximum(std, 1e-8)
    max_abs = np.abs(scores).max()
    if max_abs < 1e-10:
        return np.zeros(env.n, dtype=np.float32)
    return (scores / max_abs).astype(np.float32)
