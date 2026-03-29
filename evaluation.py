"""
evaluation.py -- Unified rollout, evaluation, and comparison utilities.
"""

from __future__ import annotations
import numpy as np
from typing import Callable
from asset_allocation.config import ScenarioConfig
from asset_allocation.environment import PortfolioEnv
from asset_allocation.utils import executed_delta


PolicyFn = Callable[[np.ndarray, PortfolioEnv], np.ndarray]


def generate_shared_returns(config: ScenarioConfig, seed: int = None) -> np.ndarray:
    """Pre-generate returns for fair cross-solver comparison.
    Shape: (n_eval_episodes, T, n)."""
    if seed is None:
        seed = config.seed
    rng = np.random.default_rng(seed)
    return rng.normal(
        loc=np.array(config.a_k),
        scale=np.sqrt(np.array(config.s_k)),
        size=(config.n_eval_episodes, config.T, config.n),
    )


def rollout_episode(env, policy_fn, shared_returns=None, seed=42):
    """Run one episode, recording full trajectory including executed deltas.

    Returns dict with keys:
        timesteps, wealth, weights, actions, executed_deltas,
        rebalanced_weights, terminal_utility
    """
    obs, _ = env.reset(seed=seed)
    traj = {
        "timesteps": [0], "wealth": [float(env.wealth)],
        "weights": [env.p.copy()], "actions": [],
        "executed_deltas": [], "rebalanced_weights": [],
    }
    for t in range(env.T):
        action = np.asarray(policy_fn(obs, env), dtype=np.float32)

        # Compute executed delta before step modifies state
        delta_exec, rebalanced = executed_delta(env, env.p, action)
        traj["actions"].append(action.copy())
        traj["executed_deltas"].append(delta_exec.copy())
        traj["rebalanced_weights"].append(rebalanced.copy())

        if shared_returns is not None:
            env._shared_return = shared_returns[t].copy()
        obs, reward, done, _, _ = env.step(action)
        traj["timesteps"].append(t + 1)
        traj["wealth"].append(float(env.wealth))
        traj["weights"].append(env.p.copy())
    traj["terminal_utility"] = float(reward)
    return traj


def evaluate_policy(env, policy_fn, n_episodes=2000, seed=42, shared_returns=None):
    """Run many episodes, compute summary statistics with per-step data.

    Returns dict with keys:
        mean_utility, std_utility, mean_wealth, std_wealth,
        wealth_paths (n_ep, T+1), weight_paths (n_ep, T+1, n+1),
        delta_paths (n_ep, T, n), terminal_wealths, terminal_utilities
    """
    T, n = env.T, env.n
    utilities, terminal_wealths = [], []
    wealth_paths = np.zeros((n_episodes, T + 1))
    weight_paths = np.zeros((n_episodes, T + 1, n + 1))
    delta_paths = np.zeros((n_episodes, T, n))

    for ep in range(n_episodes):
        np.random.seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)
        wealth_paths[ep, 0] = env.wealth
        weight_paths[ep, 0] = env.p.copy()

        for t in range(T):
            action = np.asarray(policy_fn(obs, env), dtype=np.float32)

            # Compute executed delta
            d_exec, _ = executed_delta(env, env.p, action)
            delta_paths[ep, t] = d_exec

            if shared_returns is not None:
                env._shared_return = shared_returns[ep, t].copy()
            obs, reward, done, _, _ = env.step(action)
            wealth_paths[ep, t + 1] = env.wealth
            weight_paths[ep, t + 1] = env.p.copy()

        utilities.append(float(reward))
        terminal_wealths.append(float(env.wealth))

    utilities = np.array(utilities)
    terminal_wealths = np.array(terminal_wealths)

    return {
        "mean_utility": float(utilities.mean()),
        "std_utility": float(utilities.std()),
        "mean_wealth": float(terminal_wealths.mean()),
        "std_wealth": float(terminal_wealths.std()),
        "wealth_paths": wealth_paths,
        "weight_paths": weight_paths,
        "delta_paths": delta_paths,
        "terminal_wealths": terminal_wealths,
        "terminal_utilities": utilities,
    }


def format_result(result: dict, method_name: str) -> str:
    return (f"  {method_name:22s}  "
            f"utility={result['mean_utility']:.5f} +/- {result['std_utility']:.5f}  "
            f"wealth={result['mean_wealth']:.5f} +/- {result['std_wealth']:.5f}")
