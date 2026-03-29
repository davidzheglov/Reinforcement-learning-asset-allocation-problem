"""
tabular_solver.py -- Exact DP via backward induction on discretized grids.

Method:
    V_T(w) = u(w)   for all grid points
    V_t(w, p) = max_a  E[ V_{t+1}(w', p') | w, p, a ]   via Gauss-Hermite quadrature

State grid: wealth x proportion^n
Action grid: feasible deltas pre-filtered by turnover
Expectations: Gauss-Hermite quadrature (near-exact for Gaussian returns)
V_t: stored on grid, interpolated via RegularGridInterpolator

Pros: deterministic, no NN error, exact expectations
Cons: exponential grid growth (curse of dimensionality), coarse for n>3

With prop_step=0.20 and n=3: 6^3 = 216 proportion combos x 20 wealth points = 4320 states.
This is feasible but already at the edge for n=4.
"""

from __future__ import annotations
import itertools
import time
import numpy as np
from scipy.special import roots_hermite
from scipy.interpolate import RegularGridInterpolator

from asset_allocation.config import ScenarioConfig
from asset_allocation.utils import cara_utility


def solve(config: ScenarioConfig, verbose: bool = True) -> dict:
    """Solve via tabular backward induction. Returns solver_state dict."""
    n, T, r, A = config.n, config.T, config.r, config.A
    a_k = np.array(config.a_k)
    s_k = np.array(config.s_k)

    # Gauss-Hermite quadrature
    nodes_1d, weights_1d = roots_hermite(config.n_quad)
    weights_1d_norm = weights_1d / np.sqrt(np.pi)

    asset_returns = []
    for k in range(n):
        asset_returns.append(np.sqrt(2 * s_k[k]) * nodes_1d + a_k[k])

    grids = np.meshgrid(*asset_returns, indexing='ij')
    joint_returns = np.stack([g.ravel() for g in grids], axis=1)
    wgrids = np.meshgrid(*[weights_1d_norm] * n, indexing='ij')
    joint_weights = np.prod(np.stack([g.ravel() for g in wgrids], axis=1), axis=1)
    Q = len(joint_weights)

    # State grids
    w_grid = np.linspace(config.wealth_min, config.wealth_max, config.wealth_points)
    p_grid = np.arange(config.prop_min, config.prop_max + config.prop_step * 0.5, config.prop_step)
    p_grid = np.round(p_grid, 6)
    n_pg = len(p_grid)

    all_p_indices = list(itertools.product(range(n_pg), repeat=n))
    all_p_values = np.array([[p_grid[i] for i in idx] for idx in all_p_indices])

    # Action grid (feasible deltas)
    act_vals = np.arange(-config.action_max, config.action_max + config.action_step * 0.5,
                         config.action_step)
    act_vals = np.round(act_vals, 6)
    all_actions = np.array(list(itertools.product(act_vals, repeat=n)))
    d_cash = -all_actions.sum(axis=1)
    turnover = 0.5 * (np.abs(d_cash) + np.abs(all_actions).sum(axis=1))
    valid_actions = all_actions[turnover <= config.max_turnover + 1e-10]

    if verbose:
        print(f"[Tabular DP] n={n}, T={T}, r={r}, A={A}")
        print(f"  Wealth: {len(w_grid)} pts, Prop: {n_pg} pts, "
              f"Combos: {len(all_p_indices)}, Actions: {len(valid_actions)}, Quad: {Q}")

    grid_shape = (len(w_grid),) + (n_pg,) * n
    v_grids = {}
    policy = {}

    # Terminal
    v_grids[T] = np.zeros(grid_shape)
    for wi, W in enumerate(w_grid):
        v_grids[T][wi] = cara_utility(W, A)

    if verbose:
        print(f"\n{'=' * 60}\nBackward Induction (Tabular)\n{'=' * 60}")
    total_start = time.time()

    for t in reversed(range(T)):
        step_start = time.time()
        interp = RegularGridInterpolator(
            (w_grid,) + (p_grid,) * n, v_grids[t + 1],
            method='linear', bounds_error=False, fill_value=None,
        )

        V_t = np.full(grid_shape, -1e20)
        pol_t = np.zeros(grid_shape + (n,))
        n_computed = 0

        for wi, w in enumerate(w_grid):
            for pi, pidx in enumerate(all_p_indices):
                p_risky = all_p_values[pi]
                p_cash = 1.0 - p_risky.sum()

                if abs(p_cash) + abs(p_risky).sum() > config.leverage_factor + 1.0:
                    V_t[(wi,) + pidx] = cara_utility(0.01, A)
                    continue

                new_p = p_risky + valid_actions
                new_cash = 1.0 - new_p.sum(axis=1)

                gross = np.abs(new_cash) + np.abs(new_p).sum(axis=1)
                fidx = np.where(gross <= config.leverage_factor + 1e-10)[0]
                if len(fidx) == 0:
                    V_t[(wi,) + pidx] = cara_utility(0.01, A)
                    continue

                new_p_f = new_p[fidx]
                new_cash_f = new_cash[fidx]
                F = len(fidx)

                port_ret = new_cash_f[:, None] * r + new_p_f @ joint_returns.T
                w_next = w * (1.0 + port_ret)

                denom = 1.0 + port_ret
                p_next = (new_p_f[:, None, :] *
                          (1.0 + joint_returns[None, :, :])) / denom[:, :, None]

                w_clip = np.clip(w_next, w_grid[0], w_grid[-1])
                p_clip = np.clip(p_next, p_grid[0], p_grid[-1])
                pts = np.column_stack([w_clip.ravel(), p_clip.reshape(F * Q, n)])
                vals = interp(pts)

                bankrupt = w_next.ravel() <= 0
                if np.any(bankrupt):
                    vals[bankrupt] = cara_utility(0.001, A)

                evs = vals.reshape(F, Q) @ joint_weights
                best = np.argmax(evs)

                V_t[(wi,) + pidx] = evs[best]
                pol_t[(wi,) + pidx] = valid_actions[fidx[best]]
                n_computed += 1

        v_grids[t] = V_t
        policy[t] = pol_t
        if verbose:
            print(f"  t={t}: {n_computed:,} states, {time.time() - step_start:.1f}s")

    if verbose:
        print(f"Total: {time.time() - total_start:.1f}s")

    # Build policy interpolators
    policy_interps = {}
    for ts in range(T):
        interps = []
        for k in range(n):
            interps.append(RegularGridInterpolator(
                (w_grid,) + (p_grid,) * n, policy[ts][..., k],
                method='linear', bounds_error=False, fill_value=None,
            ))
        policy_interps[ts] = interps

    return {
        "v_grids": v_grids, "policy": policy, "policy_interps": policy_interps,
        "config": {"n": n, "T": T, "w_grid": w_grid, "p_grid": p_grid},
    }


def get_policy_fn(solver_state: dict):
    """Create policy function: (obs, env) -> action in [-1,1]^n."""
    interps = solver_state["policy_interps"]
    sc = solver_state["config"]
    w_grid, p_grid, n = sc["w_grid"], sc["p_grid"], sc["n"]

    def policy_fn(obs, env):
        t = min(int(round(obs[0] * env.T)), env.T - 1)
        W = np.clip(obs[1], w_grid[0], w_grid[-1])
        p_risky = np.clip(obs[3:], p_grid[0], p_grid[-1])
        pt = np.concatenate([[W], p_risky]).reshape(1, -1)
        delta = np.array([interps[t][k](pt)[0] for k in range(n)])
        return np.clip(delta / env.action_max, -1.0, 1.0).astype(np.float32)

    return policy_fn
