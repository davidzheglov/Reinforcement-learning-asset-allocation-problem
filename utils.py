"""
utils.py -- Shared helper functions used by multiple solvers.

Contains:
    - CARA utility function
    - Feasible action grid builder
    - State sampling for ADP
"""

from __future__ import annotations
import itertools
import numpy as np
from asset_allocation.config import ScenarioConfig


def cara_utility(W: np.ndarray, A: float) -> np.ndarray:
    """CARA (exponential) utility: u(W) = (1 - exp(-A*W)) / A."""
    return (1.0 - np.exp(-A * np.clip(W, -20, 100))) / A


def build_feasible_actions(config: ScenarioConfig, step: float = None) -> np.ndarray:
    """
    Build discrete action set in TRUE DELTA SPACE with turnover pre-filtering.

    Each action is (delta_1, ..., delta_n) where delta_k is the change to the
    risky weight of asset k.  Cash delta is implied: delta_cash = -sum(delta_risky).

    Turnover constraint:  0.5 * (|delta_cash| + sum|delta_k|) <= max_turnover
    This is a JOINT constraint -- it couples all deltas together.

    We do NOT filter by state-dependent constraints here (leverage, long-only)
    because those depend on the current portfolio weights.

    With step=0.025 and action_max=0.10, per-asset grid is:
        [-0.10, -0.075, -0.05, -0.025, 0, 0.025, 0.05, 0.075, 0.10]

    Returns
    -------
    actions : ndarray (n_feasible, n)  -- each row is a delta vector
    """
    if step is None:
        step = config.adp_action_step
    amax = config.action_max

    vals = np.arange(-amax, amax + step * 0.5, step)
    vals = np.round(vals, 8)

    all_actions = np.array(list(itertools.product(vals, repeat=config.n)))
    delta_cash = -all_actions.sum(axis=1)
    turnover = 0.5 * (np.abs(delta_cash) + np.abs(all_actions).sum(axis=1))
    feasible = turnover <= config.max_turnover + 1e-10

    return all_actions[feasible]


def check_feasibility(
    p_risky: np.ndarray,
    actions: np.ndarray,
    config: ScenarioConfig,
) -> tuple:
    """
    Apply actions to states and check state-dependent feasibility.

    Parameters
    ----------
    p_risky : (S, n) current risky proportions
    actions : (A, n) candidate deltas

    Returns
    -------
    p_new : (S, A, n) new risky proportions
    new_cash : (S, A) new cash proportions
    feasible : (S, A) bool mask
    """
    p_new = p_risky[:, None, :] + actions[None, :, :]
    new_cash = 1.0 - p_new.sum(axis=2)

    if config.allow_short:
        gross = np.abs(new_cash) + np.abs(p_new).sum(axis=2)
        feasible = gross <= config.leverage_factor + 1e-10
    else:
        all_nonneg = np.all(p_new >= -1e-10, axis=2) & (new_cash >= -1e-10)
        total_risky = p_new.sum(axis=2)
        feasible = all_nonneg & (total_risky <= 1.0 + 1e-10)

    return p_new, new_cash, feasible


def sample_reachable_states(
    config: ScenarioConfig,
    t: int,
    actions: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = None,
) -> tuple:
    """
    Sample reachable (wealth, risky_proportions) at time t by rolling forward
    from p_init with random feasible actions.  Vectorized.

    Returns (W, p_risky) of shapes (n_samples,), (n_samples, n).
    """
    n = config.n
    if n_samples is None:
        n_samples = config.adp_n_train
    a_k = np.array(config.a_k)
    std_k = config.std_k
    r = config.r
    p_init_risky = np.array(config.p_init)[1:]

    W = np.ones(n_samples)
    p = np.tile(p_init_risky, (n_samples, 1))
    n_actions = len(actions)

    for step in range(t):
        idx = rng.integers(0, n_actions, size=n_samples)
        deltas = actions[idx]
        p_new = p + deltas
        new_cash = 1.0 - p_new.sum(axis=1)

        # Enforce constraints
        if config.allow_short:
            gross = np.abs(new_cash) + np.abs(p_new).sum(axis=1)
            over = gross > config.leverage_factor + 1e-10
            if np.any(over):
                scale = config.leverage_factor / np.maximum(gross[over], 1e-12)
                p_new[over] = p[over] + deltas[over] * scale[:, None]
                new_cash = 1.0 - p_new.sum(axis=1)
        else:
            p_new = np.clip(p_new, 0.0, 1.0)
            total = p_new.sum(axis=1)
            over = total > 1.0
            if np.any(over):
                p_new[over] /= total[over, None]
            new_cash = 1.0 - p_new.sum(axis=1)

        R = rng.normal(a_k, std_k, size=(n_samples, n))
        port_ret = new_cash * r + (p_new * R).sum(axis=1)
        W_new = W * (1.0 + port_ret)
        denom = 1.0 + port_ret
        p_new_drifted = p_new * (1.0 + R) / denom[:, None]

        bankrupt = W_new <= 0.001
        p_new_drifted[bankrupt] = p_init_risky
        W_new[bankrupt] = 0.001

        W = W_new
        p = p_new_drifted

    return W, p


def executed_delta(env, p_full, action):
    """
    Compute the actual executed risky-weight delta from a raw policy action.

    Replicates the constraint enforcement in PortfolioEnv.step() before returns
    are applied: scaling by action_max, turnover projection, and long-only or
    leverage clipping.  This is the single source of truth for "what trade
    actually happens" and must be used everywhere decisions are plotted or
    compared.

    Parameters
    ----------
    env : PortfolioEnv
        Environment instance (provides action_max, max_turnover, etc.)
    p_full : array (n+1,)
        Current portfolio weights [cash, p1, ..., pn]
    action : array (n,)
        Raw action in [-1, 1]^n

    Returns
    -------
    delta_exec : np.ndarray (n,)
        Actual executed change in risky weights
    rebalanced : np.ndarray (n+1,)
        Post-rebalance portfolio weights (before returns)
    """
    p_full = np.asarray(p_full, dtype=np.float64)
    action = np.asarray(action, dtype=np.float64)

    delta_risky = action * env.action_max
    delta_cash = -delta_risky.sum()

    turnover = 0.5 * (abs(delta_cash) + np.abs(delta_risky).sum())
    if turnover > env.max_turnover + 1e-10:
        scale = env.max_turnover / turnover
        delta_risky *= scale
        delta_cash = -delta_risky.sum()

    new_risky = p_full[1:] + delta_risky
    new_cash = p_full[0] + delta_cash

    if env.allow_short:
        gross = abs(new_cash) + np.abs(new_risky).sum()
        if gross > env.leverage_factor + 1e-10:
            s = env.leverage_factor / gross
            new_risky *= s
            new_cash = 1.0 - new_risky.sum()
    else:
        new_risky = np.clip(new_risky, 0.0, 1.0)
        new_cash = np.clip(new_cash, 0.0, 1.0)
        total = new_risky.sum()
        if total > 1.0:
            new_risky /= total
            total = 1.0
        new_cash = 1.0 - total

    rebalanced = np.concatenate([[new_cash], new_risky])
    delta_exec = rebalanced[1:] - p_full[1:]
    return delta_exec, rebalanced
