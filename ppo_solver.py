"""
ppo_solver.py -- PPO (Proximal Policy Optimization) via stable-baselines3.

Model-free approach: does not exploit known Gaussian return structure.
More sample-inefficient than DP methods but robust to model misspecification.

Uses gamma=1.0 (no discounting) since the problem is finite-horizon with
terminal reward only.

Caching note: SB3 models cannot be pickled via torch.save (especially in
Colab where the kernel class is un-picklable).  We use SB3's native
model.save() / PPO.load() which serialize only the policy weights +
hyperparameters, avoiding environment references entirely.
"""

from __future__ import annotations
import time
import numpy as np
from pathlib import Path

from asset_allocation.config import ScenarioConfig
from asset_allocation.environment import PortfolioEnv
from asset_allocation.caching import cache_path, config_hash

CACHE_DIR = Path(__file__).parent / "outputs" / "cache"


def _ppo_cache_path(config: ScenarioConfig) -> Path:
    """Return cache path for PPO (uses .zip extension, SB3 convention)."""
    safe_name = config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    h = config_hash(config, "ppo")
    return CACHE_DIR / f"ppo_{safe_name}_{h}"


def solve(config: ScenarioConfig, verbose: bool = True) -> dict:
    """Train PPO on the portfolio environment."""
    from stable_baselines3 import PPO

    # Check cache (SB3 saves as .zip)
    cp = _ppo_cache_path(config)
    zip_path = Path(str(cp) + ".zip")
    if config.cache_enabled and zip_path.exists():
        if verbose:
            print(f"  [ppo] Loading from cache: {zip_path.name}")
        env = PortfolioEnv(config)
        model = PPO.load(str(cp), env=env, device="cpu")
        return {"model": model, "config": {"n": config.n, "T": config.T}}

    if verbose:
        print(f"[PPO] n={config.n}, T={config.T}, A={config.A}, "
              f"timesteps={config.ppo_timesteps}")

    env = PortfolioEnv(config)
    start = time.time()

    model = PPO(
        "MlpPolicy", env,
        learning_rate=config.ppo_lr,
        gamma=1.0,
        n_steps=2048,
        batch_size=64,
        verbose=1 if verbose else 0,
        seed=config.seed,
        device="cpu",      # MlpPolicy runs faster on CPU than GPU
    )
    model.learn(total_timesteps=config.ppo_timesteps)

    if verbose:
        print(f"[PPO] Training: {time.time() - start:.1f}s")

    # Cache using SB3 native save (avoids pickle issues in Colab/Jupyter)
    if config.cache_enabled:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        model.save(str(cp))
        if verbose:
            print(f"  [ppo] Cached to: {zip_path.name}")

    return {"model": model, "config": {"n": config.n, "T": config.T}}


def get_policy_fn(solver_state: dict):
    """Create policy function: (obs, env) -> action in [-1,1]^n."""
    model = solver_state["model"]

    def policy_fn(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    return policy_fn
