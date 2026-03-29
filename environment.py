"""
environment.py -- Gymnasium environment for constrained multi-asset allocation.

State:  [time/T, wealth, p_cash, p_1, ..., p_n]
Action: [a_1, ..., a_n] in [-1, 1]^n, scaled by action_max to get deltas
Reward: 0 at non-terminal, CARA utility u(W_T) at terminal

Supports long-only and long-short (leverage) via ScenarioConfig.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from asset_allocation.config import ScenarioConfig


class PortfolioEnv(gym.Env):
    """Gymnasium environment for constrained multi-asset portfolio allocation."""

    metadata = {"render_modes": []}

    def __init__(self, config: ScenarioConfig = None, **kwargs):
        super().__init__()

        if config is not None:
            n, T, r, A = config.n, config.T, config.r, config.A
            a_k, s_k = config.a_k, config.s_k
            p_init = config.p_init
            max_turnover = config.max_turnover
            leverage_factor = config.leverage_factor
            prop_min = config.prop_min
            action_max = config.action_max
        else:
            n = kwargs.get("n", 3)
            T = kwargs.get("T", 5)
            r = kwargs.get("r", 0.03)
            A = kwargs.get("A", 0.5)
            a_k = kwargs.get("a_k", [0.08, 0.06, 0.10][:n])
            s_k = kwargs.get("s_k", [0.02, 0.015, 0.04][:n])
            p_init = kwargs.get("p_init", [1.0 / (n + 1)] * (n + 1))
            max_turnover = kwargs.get("max_turnover", 0.10)
            leverage_factor = kwargs.get("leverage_factor", 1.0)
            prop_min = kwargs.get("prop_min", 0.0)
            action_max = kwargs.get("action_max", 0.10)

        self.n = n
        self.T = T
        self.r = r
        self.A = A
        self.a_k = np.array(a_k)
        self.s_k = np.array(s_k)
        self.max_turnover = max_turnover
        self.leverage_factor = leverage_factor
        self.allow_short = prop_min < -1e-10
        self.action_max = action_max
        self.p_init = np.array(p_init, dtype=np.float64)

        assert len(self.a_k) == n
        assert len(self.p_init) == n + 1

        self.action_space = spaces.Box(-1.0, 1.0, shape=(n,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n + 3,), dtype=np.float32)

        self.t = 0
        self.wealth = 1.0
        self.p = self.p_init.copy()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.wealth = 1.0
        self.p = self.p_init.copy()
        return self._obs(), {}

    def _obs(self):
        return np.concatenate(([self.t / self.T, self.wealth], self.p)).astype(np.float32)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        # Scale to deltas
        delta_risky = action * self.action_max
        delta_cash = -delta_risky.sum()

        # Turnover projection
        turnover = 0.5 * (abs(delta_cash) + np.abs(delta_risky).sum())
        if turnover > self.max_turnover + 1e-10:
            scale = self.max_turnover / turnover
            delta_risky *= scale
            delta_cash = -delta_risky.sum()

        new_risky = self.p[1:] + delta_risky
        new_cash = self.p[0] + delta_cash

        if self.allow_short:
            gross = abs(new_cash) + np.abs(new_risky).sum()
            if gross > self.leverage_factor + 1e-10:
                s = self.leverage_factor / gross
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

        # Sample returns (or use injected shared returns)
        if hasattr(self, '_shared_return') and self._shared_return is not None:
            R = self._shared_return
            self._shared_return = None
        else:
            R = np.random.normal(self.a_k, np.sqrt(self.s_k))

        port_ret = new_cash * self.r + (new_risky * R).sum()
        W_new = self.wealth * (1.0 + port_ret)

        if W_new <= 0.001:
            self.wealth = 0.001
            self.p = np.concatenate([[1.0], np.zeros(self.n)])
        else:
            self.wealth = W_new
            denom = 1.0 + port_ret
            self.p = np.concatenate([
                [new_cash * (1.0 + self.r) / denom],
                new_risky * (1.0 + R) / denom,
            ])

        self.t += 1
        done = self.t >= self.T
        reward = (1.0 - np.exp(-self.A * self.wealth)) / self.A if done else 0.0

        return self._obs(), float(reward), done, False, {"rebalanced_weights": rebalanced}
