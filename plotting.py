"""
plotting.py -- Visualization utilities for portfolio allocation results.

Plots:
    1. Wealth progression (mean +/- std over episodes)
    2. Terminal utility comparison (bar chart with error bars)
    3. Terminal wealth distribution (box plot)
    4. Portfolio weights over time (per-asset subplots, all methods)
    5. Rebalancing actions over time (executed deltas, per-asset subplots)
    6. Economic intuition (excess returns, Sharpe, initial executed deltas)
    7. Decision anatomy (single-episode multi-panel breakdown)

All decision-related plots use executed deltas computed through the
centralized executed_delta() helper in utils.py, which replicates the
full constraint enforcement pipeline of PortfolioEnv.step().
"""

from __future__ import annotations
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "outputs" / "figures"

METHOD_COLORS = {
    "hold": "#7f7f7f", "heuristic": "#bcbd22",
    "tabular": "#1f77b4", "adp_hermite": "#ff7f0e",
    "adp_mc": "#2ca02c", "dp_v2": "#d62728",
    "ppo": "#9467bd", "a2c": "#e377c2",
}


def _ensure_dir():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _get_color(name):
    return METHOD_COLORS.get(name, "#17becf")


def _safe_fname(scenario_name):
    return scenario_name.replace(" ", "_").replace("(", "").replace(")", "").lower()


# ======================================================================
#  1. Wealth Progression (mean +/- std)
# ======================================================================

def plot_wealth_progression(results: dict, scenario_name: str, save: bool = True):
    """Plot mean wealth paths with +/- 1 std shading."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, res in results.items():
        paths = res["wealth_paths"]  # (n_episodes, T+1)
        mean = paths.mean(axis=0)
        std = paths.std(axis=0)
        x = np.arange(len(mean))
        c = _get_color(name)
        ax.plot(x, mean, label=name, color=c, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=c)

    ax.set_xlabel("Time Step")
    ax.set_ylabel("Wealth")
    ax.set_title(f"Wealth Progression: {scenario_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save:
        _ensure_dir()
        fname = FIGURES_DIR / f"wealth_{_safe_fname(scenario_name)}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname.name}")
    plt.show()
    return fig


# ======================================================================
#  2. Terminal Utility Comparison (bar chart)
# ======================================================================

def plot_utility_comparison(results: dict, scenario_name: str, save: bool = True):
    """Bar chart of mean terminal utility with std error bars."""
    import matplotlib.pyplot as plt

    names = list(results.keys())
    means = [results[n]["mean_utility"] for n in names]
    stds = [results[n]["std_utility"] for n in names]
    colors = [_get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, alpha=0.8)

    ax.set_ylabel("Mean Terminal Utility")
    ax.set_title(f"Utility Comparison: {scenario_name}")
    ax.grid(True, alpha=0.3, axis='y')

    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{m:.4f}', ha='center', va='bottom', fontsize=9)

    if save:
        _ensure_dir()
        fname = FIGURES_DIR / f"utility_{_safe_fname(scenario_name)}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname.name}")
    plt.show()
    return fig


# ======================================================================
#  3. Terminal Wealth Distribution (box plot)
# ======================================================================

def plot_wealth_boxplot(results: dict, scenario_name: str, save: bool = True):
    """Box plot of terminal wealth distributions."""
    import matplotlib.pyplot as plt

    names = list(results.keys())
    data = [results[n]["terminal_wealths"] for n in names]
    colors = [_get_color(n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=names, patch_artist=True, showfliers=False)

    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    ax.set_ylabel("Terminal Wealth")
    ax.set_title(f"Terminal Wealth Distribution: {scenario_name}")
    ax.grid(True, alpha=0.3, axis='y')

    if save:
        _ensure_dir()
        fname = FIGURES_DIR / f"wealth_box_{_safe_fname(scenario_name)}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname.name}")
    plt.show()
    return fig


# ======================================================================
#  4. Portfolio Weights Over Time (averaged over episodes)
# ======================================================================

def plot_portfolio_weights(results: dict, config, scenario_name: str,
                           show_cash: bool = True, save: bool = True):
    """Portfolio weights over time, one subplot per asset (+ optional cash).

    Parameters
    ----------
    results : {method_name: eval_result_dict}  -- must contain 'weight_paths'
    config : ScenarioConfig
    """
    import matplotlib.pyplot as plt

    n = config.n
    T = config.T
    x = np.arange(T + 1)
    n_cols = (n + 1) if show_cash else n

    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 4), squeeze=False)
    labels = (["Cash"] if show_cash else []) + [f"Asset {k+1}" for k in range(n)]

    for name, res in results.items():
        wp = res["weight_paths"]  # (n_ep, T+1, n+1)
        mean_w = wp.mean(axis=0)  # (T+1, n+1)
        c = _get_color(name)

        for col_idx in range(n_cols):
            asset_idx = col_idx if show_cash else col_idx + 1
            axes[0, col_idx].plot(x, mean_w[:, asset_idx],
                                  label=name, color=c, linewidth=2)

    for col_idx in range(n_cols):
        axes[0, col_idx].set_xlabel("Time Step")
        axes[0, col_idx].set_ylabel("Weight")
        axes[0, col_idx].set_title(labels[col_idx])
        axes[0, col_idx].grid(True, alpha=0.3)
        axes[0, col_idx].legend(fontsize=7)

    fig.suptitle(f"Portfolio Weights: {scenario_name}", fontsize=12, y=1.02)
    plt.tight_layout()

    if save:
        _ensure_dir()
        fname = FIGURES_DIR / f"weights_{_safe_fname(scenario_name)}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname.name}")
    plt.show()
    return fig


# ======================================================================
#  5. Rebalancing Actions Over Time (executed deltas, averaged)
# ======================================================================

def plot_rebalancing_actions(results: dict, config, scenario_name: str,
                             save: bool = True):
    """Executed rebalancing deltas over time, one subplot per risky asset.

    Parameters
    ----------
    results : {method_name: eval_result_dict}  -- must contain 'delta_paths'
    config : ScenarioConfig
    """
    import matplotlib.pyplot as plt

    n = config.n
    T = config.T
    x = np.arange(T)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for name, res in results.items():
        dp = res["delta_paths"]  # (n_ep, T, n)
        mean_d = dp.mean(axis=0)  # (T, n)
        c = _get_color(name)

        for col in range(n):
            axes[0, col].plot(x, mean_d[:, col], label=name, color=c,
                              linewidth=2, marker='o', markersize=4)

    for col in range(n):
        axes[0, col].axhline(0, color='k', linewidth=0.5, linestyle='--')
        axes[0, col].set_xlabel("Time Step")
        axes[0, col].set_ylabel("Executed Delta")
        axes[0, col].set_title(f"Asset {col+1} Rebalancing")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=7)

    fig.suptitle(f"Executed Rebalancing Actions: {scenario_name}", fontsize=12, y=1.02)
    plt.tight_layout()

    if save:
        _ensure_dir()
        fname = FIGURES_DIR / f"rebalancing_{_safe_fname(scenario_name)}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname.name}")
    plt.show()
    return fig


# ======================================================================
#  6. Economic Intuition (excess returns, Sharpe, initial executed deltas)
# ======================================================================

def plot_economic_intuition(policy_fns: dict, config, scenario_name: str,
                            save: bool = True):
    """Show WHY the policy makes sense: excess returns, Sharpe ratios,
    and the ACTUAL EXECUTED initial rebalancing deltas for each method.

    The deltas in panel 3 go through the full constraint pipeline
    (action_max scaling, turnover projection, long-only / leverage clipping).
    """
    import matplotlib.pyplot as plt
    from asset_allocation.environment import PortfolioEnv
    from asset_allocation.utils import executed_delta

    n = config.n
    env = PortfolioEnv(config)
    excess = np.array(config.a_k) - config.r
    sharpe = excess / np.sqrt(np.array(config.s_k))
    asset_labels = [f"Asset {k+1}" for k in range(n)]

    # Initial observation
    p_init = np.array(config.p_init)
    obs_init = np.concatenate([[0.0 / config.T, 1.0], p_init]).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Excess returns
    ax = axes[0]
    colors_bar = ['green' if e > 0 else 'red' for e in excess]
    ax.bar(asset_labels, excess, color=colors_bar, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylabel("Excess Return (a_k - r)")
    ax.set_title(f"Excess Returns (r={config.r:.2f})")
    ax.grid(True, alpha=0.3, axis='y')
    for i, e in enumerate(excess):
        ax.text(i, e, f"{e:+.3f}", ha='center',
                va='bottom' if e >= 0 else 'top', fontsize=9)

    # Panel 2: Sharpe ratios
    ax = axes[1]
    colors_bar = ['green' if s > 0 else 'red' for s in sharpe]
    ax.bar(asset_labels, sharpe, color=colors_bar, alpha=0.7)
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Risk-Adjusted Attractiveness")
    ax.grid(True, alpha=0.3, axis='y')
    for i, s in enumerate(sharpe):
        ax.text(i, s, f"{s:.2f}", ha='center',
                va='bottom' if s >= 0 else 'top', fontsize=9)

    # Panel 3: Initial EXECUTED deltas from each method
    ax = axes[2]
    x = np.arange(n)
    width = 0.8 / max(len(policy_fns), 1)
    for j, (method_name, pfn) in enumerate(policy_fns.items()):
        action = np.asarray(pfn(obs_init, env), dtype=np.float64)
        delta_exec, _ = executed_delta(env, p_init, action)
        offset = (j - len(policy_fns) / 2 + 0.5) * width
        ax.bar(x + offset, delta_exec, width, label=method_name,
               color=_get_color(method_name), alpha=0.8)

    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(asset_labels)
    ax.set_ylabel("Executed Delta at t=0")
    ax.set_title("Initial Rebalancing Decision")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f"Economic Intuition: {scenario_name}", fontsize=13, y=1.02)
    plt.tight_layout()

    if save:
        _ensure_dir()
        fname = FIGURES_DIR / f"intuition_{_safe_fname(scenario_name)}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname.name}")
    plt.show()
    return fig


# ======================================================================
#  7. Decision Anatomy (single-episode multi-panel breakdown)
# ======================================================================

def plot_decision_anatomy(policy_fns: dict, config, scenario_name: str,
                          shared_returns=None, seed: int = 42,
                          save: bool = True):
    """Detailed visualization of ONE episode with shared returns across methods.

    Panels:
    - Top left:  Wealth trajectory per method
    - Top right: Cash weight over time per method
    - Bottom left:  All asset weights over time (best method)
    - Bottom right: Executed deltas at each step (best method)
    """
    import matplotlib.pyplot as plt
    from asset_allocation.environment import PortfolioEnv
    from asset_allocation.evaluation import rollout_episode

    n = config.n
    ep_returns = shared_returns if shared_returns is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    trajectories = {}
    for name, pfn in policy_fns.items():
        traj = rollout_episode(PortfolioEnv(config), pfn,
                               shared_returns=ep_returns, seed=seed)
        trajectories[name] = traj

    # --- Panel 1: Wealth ---
    ax = axes[0, 0]
    for name, traj in trajectories.items():
        ax.plot(traj["timesteps"], traj["wealth"],
                label=name, color=_get_color(name), linewidth=2)
    ax.set_ylabel("Wealth")
    ax.set_title("Wealth Trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Cash weight ---
    ax = axes[0, 1]
    for name, traj in trajectories.items():
        cash = [w[0] for w in traj["weights"]]
        ax.plot(traj["timesteps"], cash,
                label=name, color=_get_color(name), linewidth=2)
    ax.set_ylabel("Cash Weight")
    ax.set_title("Cash Allocation Over Time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Asset weights for best method ---
    ax = axes[1, 0]
    best_method = max(trajectories, key=lambda k: trajectories[k]["terminal_utility"])
    traj = trajectories[best_method]
    for k in range(n):
        w_k = [w[k + 1] for w in traj["weights"]]
        ax.plot(traj["timesteps"], w_k,
                label=f"Asset {k+1} (a={config.a_k[k]:.2f}, s={config.s_k[k]:.3f})",
                linewidth=2)
    cash = [w[0] for w in traj["weights"]]
    ax.plot(traj["timesteps"], cash, label=f"Cash (r={config.r:.2f})",
            linewidth=2, linestyle='--', color='gray')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Weight")
    ax.set_title(f"Portfolio Weights ({best_method})")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Panel 4: Executed deltas for best method ---
    ax = axes[1, 1]
    deltas = traj["executed_deltas"]
    for k in range(n):
        d_k = [d[k] for d in deltas]
        ax.plot(range(len(d_k)), d_k,
                label=f"Delta Asset {k+1}", linewidth=2, marker='o', markersize=4)
    ax.axhline(0, color='k', linewidth=0.5, linestyle='--')
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Executed Delta")
    ax.set_title(f"Executed Rebalancing Actions ({best_method})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Economic annotation ---
    excess = np.array(config.a_k) - config.r
    sharpe = excess / np.sqrt(np.array(config.s_k))
    annotation = (f"r={config.r:.2f}, A={config.A:.1f}\n"
                  f"Excess returns: {[f'{e:+.2f}' for e in excess]}\n"
                  f"Sharpe ratios: {[f'{s:.2f}' for s in sharpe]}")
    fig.text(0.5, -0.02, annotation, ha='center', fontsize=9,
             style='italic', color='gray')

    fig.suptitle(f"Decision Anatomy: {scenario_name}", fontsize=13, y=1.01)
    plt.tight_layout()

    if save:
        _ensure_dir()
        fname = FIGURES_DIR / f"anatomy_{_safe_fname(scenario_name)}.png"
        fig.savefig(str(fname), dpi=150, bbox_inches='tight')
        print(f"  Saved: {fname.name}")
    plt.show()
    return fig


# ======================================================================
#  Convenience: generate all standard plots
# ======================================================================

def plot_all(results: dict, scenario_name: str, config=None, save: bool = True):
    """Generate all standard comparison plots."""
    plot_wealth_progression(results, scenario_name, save=save)
    plot_utility_comparison(results, scenario_name, save=save)
    plot_wealth_boxplot(results, scenario_name, save=save)
    if config is not None:
        first = next(iter(results.values()))
        if "weight_paths" in first:
            plot_portfolio_weights(results, config, scenario_name, save=save)
        if "delta_paths" in first:
            plot_rebalancing_actions(results, config, scenario_name, save=save)
