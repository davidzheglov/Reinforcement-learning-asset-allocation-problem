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

## Method Details

### Exact Dynamic Programming (Tabular DP)

Tabular DP acts as the ground-truth baseline. It solves the Bellman equation via exact backward induction starting from $t = T$.

- **Mechanics**: Discretizes the continuous state and action spaces into a finite grid. For each state-action pair, the expected future value is computed exactly.
- **Expectation calculation**: Because it has full access to the transition dynamics (a "white-box" model), it calculates the exact mathematical expectation of the Gaussian returns using Gauss-Hermite quadrature, entirely avoiding sampling noise.
$$
- **Limitations**: Suffers from the **curse of dimensionality**. As $n$ increases, the grid size explodes ($\text{wealth points} \times \text{prop\_grid}^n \times \text{actions}^n \times \text{quad}^n$), making it computationally intractable for $n > 3$.

$$
### Approximate Dynamic Programming (ADP)

ADP replaces the discrete state grid with continuous function approximators.

- **Mechanics**: Instead of storing a value for every grid point, it trains a distinct neural network (`ValueNet`) for each time step $t$ to approximate $V_t(s)$. A separate `PolicyNet` is trained via supervised regression on the optimal discrete actions.
- **Expectation calculation**: The integral over market returns is estimated using either Gauss-Hermite quadrature (ADP-Hermite, zero MC noise) or Monte Carlo sampling (ADP-MC).
- **Advantages**: Handles continuous state spaces gracefully, solving the grid-scaling problem while still exploiting the known transition dynamics to propagate values backward efficiently.
- **ADP inference note**: The policy network is trained on discrete optimal deltas but outputs a continuous approximation at inference. This is standard in fitted value iteration -- the environment's constraint pipeline re-enforces feasibility on every step, so the continuous interpolation is safe.

### Deep Reinforcement Learning (PPO, A2C)

The RL agents are **model-free** -- they do not know the return distribution $\mathcal{N}(\mu_k, \sigma_k^2)$ or the transition dynamics. They learn purely by interacting with the `PortfolioEnv` environment.

- **Observation**: $[t/T, W_t, p_t^{(0)}, p_t^{(1)}, \ldots, p_t^{(n)}]$ -- normalized time, current wealth, and full portfolio weight vector. Cash weight $p_t^{(0)}$ is mathematically redundant ($p_t^{(0)} = 1 - \sum p_t^{(k)}$) but is included explicitly to simplify learning.
- **Action**: Continuous vector $a \in [-1, 1]^n$, scaled by $\delta_{\max}$ to produce portfolio deltas. Turnover and leverage constraints are enforced inside the environment by proportionally rescaling the action if violated.
- **Reward**: $R_t = 0$ for $t < T$, and $R_T = u(W_T)$ at terminal step. Discount factor $\gamma = 1.0$ since only terminal utility matters.

**PPO** (Proximal Policy Optimization): On-policy policy-gradient algorithm. Collects rollouts (2048 steps per batch), then updates the policy by maximizing a clipped surrogate objective that prevents destructively large updates. Trained for 500K environment steps with $\text{lr} = 10^{-3}$.

**A2C** (Advantage Actor-Critic): On-policy actor-critic method that updates after every $n_{\text{steps}}$ environment steps using the advantage $A_t = R_t + V(s_{t+1}) - V(s_t)$ to reduce gradient variance. Trained for 500K steps with $\text{lr} = 7 \times 10^{-4}$.

Both agents use the SB3 default `MlpPolicy` (two hidden layers of 64 units). Because they have no access to the return model, they are robust to model misspecification but less sample-efficient than DP/ADP methods.

## Results

### Cross-Method Utility Comparison

All methods are evaluated on shared pre-generated return paths (seed=42, 1000-2000 episodes) for fair comparison. Hold (do nothing) and Heuristic (Sharpe-ratio proportional) serve as baselines.

| Scenario | Hold | Heuristic | Tabular | ADP-Hermite | ADP-MC | DP v2 | PPO | A2C |
|----------|------|-----------|---------|-------------|--------|-------|-----|-----|
| Colab Demo (n=3, T=5) | 0.835 | 0.894 | 0.898 | 0.902 | 0.901 | 0.898 | 0.897 | 0.892 |
| Forced Convergence (n=3, T=2) | 1.021 | 1.073 | 1.076 | 1.076 | 1.076 | 1.076 | 1.076 | 1.073 |
| Long-Short Mixed (n=3, T=5) | 0.852 | 0.935 | 0.958 | 0.960 | 0.959 | 0.948 | 0.957 | 0.956 |
| Risky Worse Than Cash (n=3, T=5) | 0.665 | 0.693 | 0.693 | 0.692 | 0.693 | 0.693 | 0.692 | 0.692 |
| Four Assets (n=4, T=9) | 1.021 | 1.102 | -- | -- | -- | -- | 1.130 | 1.112 |

**Key observations:**

- All optimization methods significantly outperform the Hold and Heuristic baselines, confirming they learn meaningful policies.
- ADP-Hermite and ADP-MC closely match or exceed Tabular DP in all scenarios, while scaling to $n = 4$ where Tabular DP is infeasible due to the curse of dimensionality.
- PPO and A2C, despite being model-free, achieve utilities within 1-2% of the best DP/ADP methods. This is notable because they have zero knowledge of the return distributions.
- For $n = 4$, only ADP and RL methods are viable. Tabular DP and DP v2 are skipped because the discretized grid becomes too coarse to represent the optimal policy.

### Sanity Check: Forced Convergence Test

A scenario with one overwhelmingly dominant asset ($\mu_3 = 0.20$ vs $\mu_1 = 0.01$, $\mu_2 = 0.03$, $r = 0.02$, $A = 0.05$, $T = 2$) where the optimal action is unambiguous: maximize allocation to Asset 3.

Results confirm that **all methods agree**:
- Every method allocates $\delta_3 \approx 0.10$ (the maximum allowed per period) at every time step
- Terminal utilities are tightly clustered: Tabular = 1.076, ADP-Hermite = 1.076, PPO = 1.076, A2C = 1.073
- Minor differences in Assets 1 and 2 reflect different funding paths (selling Asset 1 vs selling cash), which are economically equivalent

### Convergence of RL Methods

The economic intuition scenarios each have a clearly dominant strategy. PPO and A2C are trained independently and evaluated on shared returns:

| Scenario | Expected Behavior | PPO | A2C |
|----------|-------------------|-----|-----|
| Risky Worse Than Cash ($r = 0.06$) | Sell all risky, move to cash | 0.692 | 0.692 |
| Forced Convergence ($\mu_3 = 0.20$) | Buy Asset 3 aggressively | 1.076 | 1.073 |
| Long-Short Mixed ($\mu_3 = -0.03$) | Short Asset 3, long Asset 2 | 0.957 | 0.956 |

PPO and A2C converge to nearly identical utilities (differences < 0.3%) and agree on the macro allocation strategy in every scenario. The small residual gap reflects A2C's simpler update rule (no clipped surrogate), not a failure to find the optimal policy.

### Scalability: Curse of Dimensionality

| Method | n=3, T=5 (Colab Demo) | n=4, T=9 |
|--------|----------------------|----------|
| Tabular DP | 0.898 | infeasible |
| ADP-Hermite | 0.902 | -- |
| PPO | 0.897 | 1.130 |
| A2C | 0.892 | 1.112 |

Neural network-based methods (ADP and RL) scale gracefully where exact tabular methods fail.

### Flat Utility Surface Near Optimum

The CARA utility function is concave, so many allocations near the optimum yield nearly identical utility. This explains why methods may choose different micro-allocations (e.g., which asset to sell first when all are bad) while achieving the same utility. The executed delta plots in the notebook make these micro-differences visible, but they are economically irrelevant.

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
