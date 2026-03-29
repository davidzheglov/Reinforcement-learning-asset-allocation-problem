# Multi-Asset Portfolio Allocation

Finite-horizon constrained portfolio allocation solved with dynamic programming and reinforcement learning.

## Problem

An investor allocates wealth across $n$ risky assets and a risk-free asset over $T$ periods. At each period, the investor chooses rebalancing deltas $(\delta_1, \ldots, \delta_n)$ subject to:

- **Turnover constraint**: $\frac{1}{2}(|\delta_0| + \sum |\delta_k|) \leq \tau_{\max}$
- **Long-only** or **leverage** constraints on portfolio weights
- **Terminal reward**: CARA utility $u(W_T) = (1 - e^{-AW_T}) / A$

Asset returns are i.i.d. Gaussian: $R_k \sim \mathcal{N}(\mu_k, \sigma_k^2)$.

## Methods

| Method | Type | How it works |
|--------|------|-------------|
| **Tabular DP** | Exact DP | Backward induction on discretized wealth/proportion grids with Gauss-Hermite quadrature. Gold standard for $n \leq 3$. |
| **ADP-Hermite** | Approximate DP | Neural network value/policy approximation with Gauss-Hermite expectations. Zero MC noise. |
| **ADP-MC** | Approximate DP | Same as ADP-Hermite but with Monte Carlo expectations. More flexible, some MC noise. |
| **DP v2** | Approximate DP | Normalized action grid in $[-1,1]^n$. Legacy approach, less efficient but simple. |
| **PPO** | Model-free RL | Proximal Policy Optimization via stable-baselines3. Does not exploit Gaussian structure. |
| **A2C** | Model-free RL | Advantage Actor-Critic via stable-baselines3. Synchronous updates, often competitive with PPO. |
| **Hold** | Baseline | Zero action -- keep current allocation unchanged. |
| **Heuristic** | Baseline | Allocate proportional to Sharpe ratio. |

All methods share the same `PortfolioEnv` environment with identical constraint enforcement.

## Why the Solution is Correct

### Environment dynamics

The environment (`environment.py`) implements a precise constraint enforcement pipeline on every step:

1. **Scale**: raw action $a \in [-1,1]^n$ is multiplied by `action_max` to get candidate deltas
2. **Turnover projection**: if half-turnover exceeds `max_turnover`, all deltas are proportionally scaled down
3. **Constraint clipping**: long-only mode clips weights to $[0, 1]$; leverage mode enforces gross exposure $\leq L$
4. **Return sampling**: Gaussian returns applied to rebalanced weights

### Executed deltas

The **executed delta** is the actual change in risky weights after the full constraint pipeline. This is the correct object for comparing methods because:
- Raw network outputs differ across methods (some output deltas, some output normalized actions)
- Only the executed delta reflects what trade actually happens
- The centralized `executed_delta()` function in `utils.py` replicates the environment pipeline exactly

### Economic intuition validation

The economic intuition tests verify that policies are sensible:

- **Risky worse than cash**: all methods sell risky assets (negative deltas), moving toward cash
- **Dominant asset**: all methods aggressively buy the best risk-adjusted asset
- **High risk aversion**: all methods allocate conservatively, keeping most wealth in cash
- **Long-short**: methods short the negative-return asset and go long on winners

When multiple independently-trained methods agree on the direction and magnitude of trades across diverse scenarios, this provides strong evidence of correctness.

### Cross-method agreement

- **Tabular DP** is exact (for its grid resolution) and serves as ground truth for $n \leq 3$
- **ADP-Hermite** uses exact expectations (zero MC noise) with NN approximation
- **PPO/A2C** are model-free and learn from scratch without knowing the return distribution
- Agreement between DP methods and RL methods is particularly strong validation

## How to Run on Google Colab

### Folder structure

```
/content/
  asset_allocation/
    __init__.py
    config.py
    environment.py
    utils.py
    caching.py
    evaluation.py
    plotting.py
    baselines.py
    tabular_solver.py
    adp_mc_solver.py
    adp_hermite_solver.py
    dp_v2_solver.py
    ppo_solver.py
    a2c_solver.py
    run_single.py
    run_comparison.py
    clean.py
    outputs/
      cache/       <- trained models cached here
      figures/     <- generated plots saved here
      reports/     <- JSON comparison reports
  notebooks/
    asset_allocation_demo.ipynb
```

### Steps

1. Upload the `asset_allocation/` folder to `/content/asset_allocation/`
2. Open `notebooks/asset_allocation_demo.ipynb`
3. Run the Colab setup cell (installs `stable-baselines3`, adds `/content` to path)
4. Run all cells top-to-bottom
5. Enable autoreload: the setup cell runs `%load_ext autoreload` and `%autoreload 2`

### Caching

- Trained models are cached to `outputs/cache/` as `.pt` or `.zip` files
- Cache key = SHA-256 hash of all training-relevant config parameters
- If any parameter changes, a new model is trained automatically
- To force retrain: set `config.cache_enabled = False` or delete cache files
- PPO/A2C use SB3 native `.zip` format; all others use `torch.save`

### Regenerating results

```python
# Clear all caches
from asset_allocation.clean import clean
clean("all")

# Or just clear cache
clean("cache")
```

Then re-run the notebook from the top.

## What Plots to Look At

### Wealth progression
Mean wealth over time with $\pm 1$ std shading. Shows whether methods grow wealth at similar rates.

### Terminal utility comparison
Bar chart of mean CARA utility at terminal time. The primary performance metric.

### Portfolio weights over time
One subplot per asset (+ cash). Shows the average allocation at each timestep. Useful for verifying that methods agree on *which* assets to hold and in what proportion.

### Executed rebalancing actions over time
One subplot per risky asset. Shows the average **actual executed delta** at each timestep. This is the most informative plot for understanding what each method actually *does*. Key things to look for:
- Direction: are methods buying or selling each asset?
- Magnitude: are methods trading aggressively or cautiously?
- Agreement: do DP and RL methods make similar trades?

### Economic intuition figures
Three-panel plots showing excess returns, Sharpe ratios, and initial executed deltas. Validates that the learned policy is economically rational.

### Decision anatomy
Four-panel single-episode breakdown showing wealth, cash weight, asset weights, and executed deltas for the best method.

## CLI Usage

```bash
# Run single solver
python -m asset_allocation.run_single --scenario colab_demo --solver tabular --plot

# Compare all solvers
python -m asset_allocation.run_comparison --scenario long_only_attractive --plot

# Compare specific solvers
python -m asset_allocation.run_comparison --scenario colab_demo --only hold tabular adp_hermite ppo a2c

# Run all scenarios
python -m asset_allocation.run_comparison --all-scenarios --save-report

# Clean outputs
python -m asset_allocation.clean --what all
```

## Scenarios

See `config.py` for the full list. Key scenarios:

| Scenario | Assets | Horizon | Key Feature |
|----------|--------|---------|-------------|
| `colab_demo` | 3 | 5 | Quick demo, reduced training |
| `long_only_attractive` | 3 | 5 | All assets beat cash |
| `long_only_risky_worse` | 3 | 5 | All assets worse than cash |
| `forced_convergence_test` | 3 | 2 | One dominant asset, low risk aversion |
| `high_risk_aversion` | 3 | 5 | A=3.0, conservative |
| `long_short_mixed` | 3 | 5 | Leverage + negative-return asset |
| `four_assets_T7` | 4 | 7 | Larger problem (no tabular) |

For $n=4$ scenarios, tabular DP and DP v2 are skipped (curse of dimensionality).

## Dependencies

- numpy, scipy, matplotlib
- torch (PyTorch)
- gymnasium
- stable-baselines3 (for PPO, A2C)
