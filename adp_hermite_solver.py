"""
adp_hermite_solver.py -- Approximate DP with Gauss-Hermite quadrature.

Same structure as adp_mc_solver but expectations are computed via
Gauss-Hermite quadrature instead of Monte Carlo.  This gives exact
integration for polynomial functions of Gaussian returns (zero MC noise).

Quadrature:
    integral f(x) * N(mu, sigma^2) dx
    ≈ sum_q  w_q * f(sqrt(2*sigma^2) * x_q + mu)
    where (x_q, w_q) are Hermite quadrature nodes/weights.

For n assets:  Q = n_quad^n joint quadrature points.
"""

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.special import roots_hermite

from asset_allocation.config import ScenarioConfig
from asset_allocation.utils import (
    cara_utility, build_feasible_actions, check_feasibility,
    sample_reachable_states,
)
from asset_allocation.caching import load_cache, save_cache


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def _build_quadrature(config: ScenarioConfig):
    """Build joint Gauss-Hermite quadrature grid."""
    n = config.n
    a_k = np.array(config.a_k)
    s_k = np.array(config.s_k)
    n_quad = config.adp_n_quad

    nodes_1d, weights_1d = roots_hermite(n_quad)
    weights_1d_norm = weights_1d / np.sqrt(np.pi)

    asset_returns = []
    for k in range(n):
        asset_returns.append(np.sqrt(2 * s_k[k]) * nodes_1d + a_k[k])

    grids = np.meshgrid(*asset_returns, indexing='ij')
    joint_returns = np.stack([g.ravel() for g in grids], axis=1)  # (Q, n)

    wgrids = np.meshgrid(*[weights_1d_norm] * n, indexing='ij')
    joint_weights = np.prod(
        np.stack([g.ravel() for g in wgrids], axis=1), axis=1
    )  # (Q,)

    return joint_returns, joint_weights


def solve(config: ScenarioConfig, verbose: bool = True) -> dict:
    """Solve via ADP backward induction with Gauss-Hermite expectations."""
    cached = load_cache(config, "adp_hermite", verbose=verbose)
    if cached is not None:
        return cached

    n, T, r, A = config.n, config.T, config.r, config.A
    hidden = config.adp_hidden
    n_train = config.adp_n_train
    epochs = config.adp_epochs
    lr = config.adp_lr
    rng = np.random.default_rng(config.seed)

    actions = build_feasible_actions(config, step=config.adp_action_step)
    n_act = len(actions)

    joint_returns, joint_weights = _build_quadrature(config)
    Q = len(joint_weights)

    if verbose:
        print(f"[ADP-Hermite] n={n}, T={T}, A={A}, states={n_train}, "
              f"actions={n_act}, quad={Q}")

    input_dim = 1 + n
    value_nets = {}
    policy_nets = {}
    v_stats = {}

    if verbose:
        print(f"\n{'=' * 60}\nBackward Induction (ADP-Hermite)\n{'=' * 60}")
    total_start = time.time()

    for t in reversed(range(T)):
        step_start = time.time()

        W, p_risky = sample_reachable_states(config, t, actions, rng, n_train)
        S = len(W)

        p_new, new_cash, feasible = check_feasibility(p_risky, actions, config)

        # Memory budget for chunking
        mem_budget = 200_000_000
        per_state = n_act * Q * 8 * 3
        chunk_size = max(1, mem_budget // per_state)

        targets = np.full(S, -1e20)
        best_actions = np.zeros((S, n))

        for s_start in range(0, S, chunk_size):
            s_end = min(s_start + chunk_size, S)
            cs = s_end - s_start

            p_new_c = p_new[s_start:s_end]
            nc_c = new_cash[s_start:s_end]
            feas_c = feasible[s_start:s_end]
            W_c = W[s_start:s_end]

            # Portfolio return: (cs, A, Q)
            port_ret = (nc_c[:, :, None] * r +
                        np.einsum('san,qn->saq', p_new_c, joint_returns))
            w_next = W_c[:, None, None] * (1.0 + port_ret)

            if t == T - 1:
                vals = cara_utility(w_next, A)
            else:
                denom = 1.0 + port_ret
                p_next = (p_new_c[:, :, None, :] *
                          (1.0 + joint_returns[None, None, :, :])) / denom[:, :, :, None]

                w_clip = np.clip(w_next, 0.001, 100.0)
                p_clip = np.clip(p_next, 0.0, 1.0)

                flat_w = w_clip.reshape(-1, 1)
                flat_p = p_clip.reshape(-1, n)
                nn_input = np.concatenate([flat_w, flat_p], axis=1)

                with torch.no_grad():
                    inp_t = torch.tensor(nn_input, dtype=torch.float32)
                    raw = value_nets[t + 1](inp_t).numpy()
                    vals_flat = raw * v_stats[t + 1][1] + v_stats[t + 1][0]
                vals = vals_flat.reshape(cs, n_act, Q)

                bankrupt = w_next <= 0.001
                if np.any(bankrupt):
                    vals[bankrupt] = cara_utility(0.001, A)

            # Weighted sum over quadrature points: (cs, A)
            ev = np.einsum('saq,q->sa', vals, joint_weights)

            ev[~feas_c] = -1e20

            best_idx = np.argmax(ev, axis=1)
            targets[s_start:s_end] = ev[np.arange(cs), best_idx]
            best_actions[s_start:s_end] = actions[best_idx]

        # Train value net
        valid = targets > -1e19
        if valid.sum() < 10:
            if verbose:
                print(f"  t={t}: WARNING only {valid.sum()} valid states")
            vnet = ValueNet(input_dim, hidden)
            pnet = PolicyNet(input_dim, n, hidden)
            value_nets[t] = vnet
            policy_nets[t] = pnet
            v_stats[t] = (0.0, 1.0)
            continue

        X_v = np.column_stack([W[valid], p_risky[valid]])
        y_v = targets[valid]
        v_mean, v_std = float(y_v.mean()), max(float(y_v.std()), 1e-8)
        y_norm = (y_v - v_mean) / v_std
        v_stats[t] = (v_mean, v_std)

        vnet = ValueNet(input_dim, hidden)
        opt_v = torch.optim.Adam(vnet.parameters(), lr=lr)
        X_t = torch.tensor(X_v, dtype=torch.float32)
        y_t = torch.tensor(y_norm, dtype=torch.float32)

        for ep in range(epochs):
            pred = vnet(X_t)
            loss = nn.functional.mse_loss(pred, y_t)
            opt_v.zero_grad()
            loss.backward()
            opt_v.step()

        value_nets[t] = vnet

        # Train policy net
        X_p = np.column_stack([W[valid], p_risky[valid]])
        y_p = best_actions[valid]

        pnet = PolicyNet(input_dim, n, hidden)
        opt_p = torch.optim.Adam(pnet.parameters(), lr=lr)
        Xp_t = torch.tensor(X_p, dtype=torch.float32)
        yp_t = torch.tensor(y_p, dtype=torch.float32)

        for ep in range(epochs):
            pred = pnet(Xp_t)
            loss = nn.functional.mse_loss(pred, yp_t)
            opt_p.zero_grad()
            loss.backward()
            opt_p.step()

        policy_nets[t] = pnet

        if verbose:
            print(f"  t={t}: {valid.sum()}/{S} valid, "
                  f"V=[{y_v.min():.3f}, {y_v.max():.3f}], "
                  f"{time.time() - step_start:.1f}s")

    if verbose:
        print(f"Total: {time.time() - total_start:.1f}s")

    result = {
        "value_nets": value_nets, "policy_nets": policy_nets,
        "v_stats": v_stats, "actions": actions,
        "config": {"n": n, "T": T},
    }
    save_cache(result, config, "adp_hermite", verbose=verbose)
    return result


def get_policy_fn(solver_state: dict):
    """Create policy function: (obs, env) -> action in [-1,1]^n."""
    policy_nets = solver_state["policy_nets"]
    n = solver_state["config"]["n"]

    def policy_fn(obs, env):
        t = min(int(round(obs[0] * env.T)), env.T - 1)
        W = float(obs[1])
        p_risky = obs[3:].astype(np.float64)

        if t not in policy_nets:
            return np.zeros(env.n, dtype=np.float32)

        inp = torch.tensor(
            np.concatenate([[W], p_risky]).reshape(1, -1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            delta = policy_nets[t](inp).numpy()[0]
        return np.clip(delta / env.action_max, -1.0, 1.0).astype(np.float32)

    return policy_fn
