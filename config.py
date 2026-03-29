"""
config.py -- Problem parameters, solver settings, and predefined scenarios.

Every solver reads from ScenarioConfig so that comparisons are fair:
same assets, same horizon, same initial portfolio, same seed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class ScenarioConfig:
    """Unified configuration for a portfolio allocation scenario."""

    # ---- Identity ----
    name: str = "default"

    # ---- Market / Problem ----
    n: int = 3                  # number of risky assets
    T: int = 5                  # investment horizon (periods)
    r: float = 0.03             # risk-free rate per period
    A: float = 0.5              # CARA risk-aversion coefficient
    a_k: List[float] = field(default_factory=lambda: [0.08, 0.06, 0.10])
    s_k: List[float] = field(default_factory=lambda: [0.02, 0.015, 0.04])
    p_init: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])

    # ---- Constraints ----
    max_turnover: float = 0.10      # max half-turnover per period
    leverage_factor: float = 1.0    # max gross exposure (1.0 = long-only)
    prop_min: float = 0.0           # min weight per risky asset
    prop_max: float = 1.0           # max weight per risky asset
    action_max: float = 0.10        # max delta magnitude per asset

    # ---- Tabular DP ----
    wealth_points: int = 20
    wealth_min: float = 0.05
    wealth_max: float = 4.0
    prop_step: float = 0.20         # finer than 0.25
    action_step: float = 0.025      # finer grid for tabular too
    n_quad: int = 5                 # quadrature points per asset

    # ---- ADP (shared by MC and Hermite) ----
    adp_n_train: int = 1000         # training states per backward step
    adp_hidden: int = 128           # hidden layer size
    adp_epochs: int = 300           # training epochs per step
    adp_lr: float = 1e-3
    adp_action_step: float = 0.025  # finer: 9 values per asset
    adp_mc_samples: int = 200       # MC samples (only for adp_mc)
    adp_n_quad: int = 5             # quadrature pts (only for adp_hermite)

    # ---- DP v2 ----
    dp_v2_action_grid: int = 5      # grid pts per dimension in [-1,1]
    dp_v2_mc_samples: int = 200
    dp_v2_state_samples: int = 1000
    dp_v2_hidden: List[int] = field(default_factory=lambda: [128, 128])
    dp_v2_epochs: int = 300
    dp_v2_lr: float = 1e-3

    # ---- PPO ----
    ppo_timesteps: int = 500_000
    ppo_lr: float = 1e-3

    # ---- A2C ----
    a2c_timesteps: int = 500_000
    a2c_lr: float = 7e-4
    a2c_n_steps: int = 5

    # ---- Evaluation ----
    seed: int = 42
    n_eval_episodes: int = 2000

    # ---- Caching ----
    cache_enabled: bool = True

    # ---- Derived ----
    @property
    def allow_short(self) -> bool:
        return self.prop_min < -1e-10

    @property
    def std_k(self) -> np.ndarray:
        return np.sqrt(np.array(self.s_k))

    @property
    def excess_returns(self) -> np.ndarray:
        return np.array(self.a_k) - self.r

    def summary(self) -> str:
        mode = "short+leverage" if self.allow_short else "long-only"
        return (f"{self.name}: n={self.n}, T={self.T}, r={self.r}, A={self.A}, "
                f"{mode}, lev={self.leverage_factor}")


# ========================================================================
# Predefined scenarios
# ========================================================================

SCENARIOS = {
    # ---- Core scenarios (n=3, T=5) ----

    "long_only_attractive": ScenarioConfig(
        name="Long-Only Attractive Risky",
        n=3, T=5, r=0.02, A=0.7,
        a_k=[0.08, 0.10, 0.12], s_k=[0.02, 0.025, 0.03],
        p_init=[0.70, 0.10, 0.10, 0.10],
        seed=123,
    ),

    "long_only_risky_worse": ScenarioConfig(
        name="Long-Only Risky Worse Than Cash",
        n=3, T=5, r=0.06, A=1.0,
        a_k=[0.01, 0.02, 0.03], s_k=[0.02, 0.03, 0.04],
        p_init=[0.10, 0.30, 0.30, 0.30],
        seed=321,
    ),

    "long_only_dominant": ScenarioConfig(
        name="Long-Only Dominant Asset",
        n=3, T=5, r=0.03, A=0.8,
        a_k=[0.05, 0.07, 0.16], s_k=[0.03, 0.03, 0.025],
        p_init=[0.25, 0.25, 0.25, 0.25],
        seed=999,
    ),

    "long_only_symmetric": ScenarioConfig(
        name="Long-Only Symmetric Assets",
        n=3, T=5, r=0.03, A=0.5,
        a_k=[0.09, 0.09, 0.09], s_k=[0.02, 0.02, 0.02],
        p_init=[0.40, 0.20, 0.20, 0.20],
        seed=2024,
    ),

    "mixed_quality": ScenarioConfig(
        name="Mixed Quality Assets",
        n=3, T=5, r=0.04, A=0.8,
        a_k=[0.05, 0.15, -0.02], s_k=[0.02, 0.03, 0.05],
        p_init=[0.25, 0.25, 0.25, 0.25],
        seed=900,
    ),

    # ---- Risk aversion extremes ----

    "high_risk_aversion": ScenarioConfig(
        name="High Risk Aversion (A=3.0)",
        n=3, T=5, r=0.03, A=3.0,
        a_k=[0.08, 0.10, 0.12], s_k=[0.02, 0.025, 0.03],
        p_init=[0.50, 0.20, 0.15, 0.15],
        seed=500,
    ),

    "low_risk_aversion": ScenarioConfig(
        name="Low Risk Aversion (A=0.1)",
        n=3, T=5, r=0.03, A=0.1,
        a_k=[0.08, 0.10, 0.12], s_k=[0.02, 0.025, 0.03],
        p_init=[0.50, 0.20, 0.15, 0.15],
        seed=501,
    ),

    # ---- Short selling / leverage ----

    "long_short_attractive": ScenarioConfig(
        name="Long-Short Attractive Risky",
        n=3, T=5, r=0.02, A=0.7,
        a_k=[0.08, 0.10, 0.12], s_k=[0.02, 0.025, 0.03],
        p_init=[0.70, 0.10, 0.10, 0.10],
        leverage_factor=2.0, prop_min=-0.5, prop_max=1.5,
        seed=123,
    ),

    "long_short_mixed": ScenarioConfig(
        name="Long-Short Mixed (Short Loser)",
        n=3, T=5, r=0.03, A=0.7,
        a_k=[0.04, 0.14, -0.03], s_k=[0.02, 0.025, 0.04],
        p_init=[0.30, 0.25, 0.25, 0.20],
        leverage_factor=2.0, prop_min=-0.5, prop_max=1.5,
        seed=1100,
    ),

    # ---- Edge cases ----

    "two_assets": ScenarioConfig(
        name="Two Assets (n=2)",
        n=2, T=5, r=0.03, A=0.7,
        a_k=[0.09, 0.14], s_k=[0.02, 0.04],
        p_init=[0.50, 0.25, 0.25],
        prop_step=0.20,
        seed=600,
    ),

    "high_volatility": ScenarioConfig(
        name="High Volatility",
        n=3, T=5, r=0.03, A=1.0,
        a_k=[0.10, 0.12, 0.15], s_k=[0.10, 0.15, 0.20],
        p_init=[0.60, 0.15, 0.15, 0.10],
        seed=800,
    ),

    "zero_rfr": ScenarioConfig(
        name="Zero Risk-Free Rate",
        n=3, T=5, r=0.0, A=0.7,
        a_k=[0.06, 0.08, 0.10], s_k=[0.02, 0.025, 0.03],
        p_init=[0.40, 0.20, 0.20, 0.20],
        seed=1300,
    ),

    "tight_turnover": ScenarioConfig(
        name="Tight Turnover (3%)",
        n=3, T=5, r=0.03, A=0.7,
        a_k=[0.08, 0.10, 0.12], s_k=[0.02, 0.025, 0.03],
        p_init=[0.70, 0.10, 0.10, 0.10],
        max_turnover=0.03, action_max=0.03, action_step=0.015,
        adp_action_step=0.015,
        seed=1200,
    ),

    "near_cash_asset": ScenarioConfig(
        name="Near-Cash Asset",
        n=3, T=5, r=0.03, A=0.7,
        a_k=[0.031, 0.10, 0.12],   # asset 1 barely beats cash
        s_k=[0.001, 0.025, 0.03],   # asset 1 very low vol
        p_init=[0.40, 0.20, 0.20, 0.20],
        seed=1400,
    ),

    "dominant_low_vol": ScenarioConfig(
        name="Dominant Low-Variance Asset",
        n=3, T=5, r=0.02, A=0.5,
        a_k=[0.06, 0.08, 0.14],
        s_k=[0.03, 0.04, 0.01],  # asset 3: high return, tiny variance
        p_init=[0.30, 0.25, 0.25, 0.20],
        seed=1500,
    ),

    "balanced_assets": ScenarioConfig(
        name="Balanced Assets (Equal Sharpe)",
        n=3, T=5, r=0.02, A=0.7,
        a_k=[0.06, 0.10, 0.14],
        s_k=[0.008, 0.032, 0.072],  # Sharpe ~= 0.45 each
        p_init=[0.40, 0.20, 0.20, 0.20],
        seed=1600,
    ),

    # ---- Longer horizons, n=4 ----

    "four_assets_T9": ScenarioConfig(
        name="4 Assets T=9",
        n=4, T=9, r=0.02, A=0.7,
        a_k=[0.06, 0.08, 0.10, 0.13],
        s_k=[0.015, 0.02, 0.025, 0.04],
        p_init=[0.40, 0.15, 0.15, 0.15, 0.15],
        ppo_timesteps=800_000,
        a2c_timesteps=800_000,
        seed=2000,
    ),

    "four_assets_T8": ScenarioConfig(
        name="4 Assets T=8",
        n=4, T=8, r=0.02, A=0.5,
        a_k=[0.07, 0.09, 0.11, 0.14],
        s_k=[0.02, 0.025, 0.03, 0.05],
        p_init=[0.40, 0.15, 0.15, 0.15, 0.15],
        ppo_timesteps=800_000,
        a2c_timesteps=800_000,
        seed=2100,
    ),

    "four_assets_T7": ScenarioConfig(
        name="4 Assets T=7",
        n=4, T=7, r=0.03, A=1.0,
        a_k=[0.05, 0.08, 0.12, 0.15],
        s_k=[0.015, 0.02, 0.03, 0.045],
        p_init=[0.40, 0.15, 0.15, 0.15, 0.15],
        a2c_timesteps=800_000,
        seed=2200,
    ),

    # ---- Quick demo (for Colab first-run) ----

    "colab_demo": ScenarioConfig(
        name="Colab Demo",
        n=3, T=5, r=0.02, A=0.7,
        a_k=[0.08, 0.10, 0.12], s_k=[0.02, 0.025, 0.03],
        p_init=[0.70, 0.10, 0.10, 0.10],
        adp_n_train=500, adp_epochs=200,
        dp_v2_state_samples=500, dp_v2_epochs=200,
        ppo_timesteps=200_000,
        a2c_timesteps=200_000,
        n_eval_episodes=1000,
        seed=123,
    ),

    "forced_convergence_test": ScenarioConfig(
        name="Forced Convergence Test",
        n=3,
        T=2,
        r=0.02,
        A=0.05,

        a_k=[0.01, 0.03, 0.20],
        s_k=[0.01, 0.01, 0.01],

        p_init=[0.90, 0.05, 0.03, 0.02],

        adp_n_train=2000,
        adp_epochs=400,
        dp_v2_state_samples=2000,
        dp_v2_epochs=400,
        ppo_timesteps=500_000,
        a2c_timesteps=500_000,

        n_eval_episodes=2000,
        seed=123,
    ),
}
