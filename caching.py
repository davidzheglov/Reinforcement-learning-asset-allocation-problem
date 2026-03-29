"""
caching.py -- Disk caching for trained solver models.

Cache key = SHA-256 hash of all config parameters that affect training.
If any parameter changes, a new model is trained.

Cache structure:
    outputs/cache/{solver}_{scenario}_{hash}.pt
"""

from __future__ import annotations
import hashlib
import json
import torch
from pathlib import Path
from asset_allocation.config import ScenarioConfig

CACHE_DIR = Path(__file__).parent / "outputs" / "cache"


def config_hash(config: ScenarioConfig, solver: str, extra: dict = None) -> str:
    """
    Deterministic hash of all training-relevant parameters.

    Parameters
    ----------
    config : ScenarioConfig
    solver : str -- solver name (included in hash to avoid collisions)
    extra : dict -- additional solver-specific parameters to hash
    """
    d = {
        "solver": solver,
        "n": config.n, "T": config.T, "r": config.r, "A": config.A,
        "a_k": tuple(round(x, 10) for x in config.a_k),
        "s_k": tuple(round(x, 10) for x in config.s_k),
        "p_init": tuple(round(x, 10) for x in config.p_init),
        "max_turnover": config.max_turnover,
        "leverage_factor": config.leverage_factor,
        "prop_min": config.prop_min, "prop_max": config.prop_max,
        "action_max": config.action_max,
        "seed": config.seed,
    }
    if extra:
        d.update(extra)
    raw = json.dumps(d, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def cache_path(config: ScenarioConfig, solver: str, extra: dict = None) -> Path:
    """Return full path for a cached model."""
    safe_name = config.name.replace(" ", "_").replace("(", "").replace(")", "").lower()
    h = config_hash(config, solver, extra)
    return CACHE_DIR / f"{solver}_{safe_name}_{h}.pt"


def load_cache(config: ScenarioConfig, solver: str, extra: dict = None, verbose: bool = True):
    """Load cached result if it exists and caching is enabled. Returns None if miss."""
    if not config.cache_enabled:
        return None
    path = cache_path(config, solver, extra)
    if path.exists():
        if verbose:
            print(f"  [{solver}] Loading from cache: {path.name}")
        return torch.load(str(path), weights_only=False)
    return None


def save_cache(result: dict, config: ScenarioConfig, solver: str,
               extra: dict = None, verbose: bool = True):
    """Save result to disk cache."""
    if not config.cache_enabled:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(config, solver, extra)
    torch.save(result, str(path))
    if verbose:
        print(f"  [{solver}] Cached to: {path.name}")
