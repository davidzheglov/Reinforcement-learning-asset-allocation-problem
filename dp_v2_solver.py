"""
dp_v2_solver.py -- DP v2: backward induction with normalized action grid + MC.

Differences from ADP solvers:
    - Actions are a grid in [-1, 1]^n, mapped to deltas via action_max
    - Expectations via Monte Carlo (like adp_mc, but different action space)
    - State sampling uses uniform random actions (not feasible deltas)

This is the "legacy" approach: simpler action parametrization but less
efficient because many grid actions may be infeasible after projection.
"""

from __future__ import annotations
import time
import itertools
import numpy as np
import torch
import torch.nn as nn

from asset_allocation.config import ScenarioConfig
from asset_allocation.utils import cara_utility
from asset_allocation.caching import load_cache, save_cache


class ValueNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _build_action_grid(n, n_grid):
    """Build normalized action grid in [-1, 1]^n."""
    vals = np.linspace(-1.0, 1.0, n_grid)
    return np.array(list(itertools.product(vals, repeat=n)))


def _apply_actions(weights, actions, env_config):
    """
    Apply normalized actions to portfolio weights.

    Parameters
    ----------
    weights : (n+1,) current weights [cash, risky_1, ..., risky_n]
    actions : (A, n) normalized in [-1, 1]^n

    Returns new_weights (A, n+1) after turnover projection and constraint enforcement.
    """
    n = env_config.n
    A = len(actions)
    action_max = env_config.action_max
    max_turnover = env_config.max_turnover

    delta_risky = actions * action_max  # (A, n)
    delta_cash = -delta_risky.sum(axis=1)  # (A,)

    # Turnover projection
    turnover = 0.5 * (np.abs(delta_cash) + np.abs(delta_risky).sum(axis=1))
    over = turnover > max_turnover + 1e-10
    if np.any(over):
        scale = max_turnover / np.maximum(turnover[over], 1e-12)
        delta_risky[over] *= scale[:, None]
        delta_cash[over] = -delta_risky[over].sum(axis=1)

    new_risky = weights[1:] + delta_risky  # (A, n)
    new_cash = weights[0] + delta_cash     # (A,)

    if env_config.allow_short:
        gross = np.abs(new_cash) + np.abs(new_risky).sum(axis=1)
        over = gross > env_config.leverage_factor + 1e-10
        if np.any(over):
            s = env_config.leverage_factor / np.maximum(gross[over], 1e-12)
            new_risky[over] *= s[:, None]
            new_cash[over] = 1.0 - new_risky[over].sum(axis=1)
    else:
        new_risky = np.clip(new_risky, 0.0, 1.0)
        total = new_risky.sum(axis=1)
        over = total > 1.0
        if np.any(over):
            new_risky[over] /= total[over, None]
        new_cash = 1.0 - new_risky.sum(axis=1)

    return np.column_stack([new_cash, new_risky])


def _sample_states(config, t, rng):
    """Sample reachable states at time t by rolling forward with random actions."""
    n = config.n
    n_states = config.dp_v2_state_samples
    a_k = np.array(config.a_k)
    std_k = config.std_k
    r = config.r

    W = np.ones(n_states)
    weights = np.tile(np.array(config.p_init), (n_states, 1))  # (S, n+1)

    for step in range(t):
        # Random normalized actions
        rand_actions = rng.uniform(-1, 1, size=(n_states, n))
        for i in range(n_states):
            new_w = _apply_actions(weights[i], rand_actions[i:i+1], config)[0]
            weights[i] = new_w

        R = rng.normal(a_k, std_k, size=(n_states, n))
        risky = weights[:, 1:]
        cash = weights[:, 0]
        port_ret = cash * r + (risky * R).sum(axis=1)
        W_new = W * (1.0 + port_ret)
        denom = 1.0 + port_ret

        new_cash = cash * (1.0 + r) / denom
        new_risky = risky * (1.0 + R) / denom[:, None]

        bankrupt = W_new <= 0.001
        W_new[bankrupt] = 0.001
        new_cash[bankrupt] = 1.0
        new_risky[bankrupt] = 0.0

        weights = np.column_stack([new_cash, new_risky])
        W = W_new

    return W, weights


def solve(config: ScenarioConfig, verbose: bool = True) -> dict:
    """Solve via DP v2 backward induction."""
    cached = load_cache(config, "dp_v2", verbose=verbose)
    if cached is not None:
        return cached

    n, T, r, A = config.n, config.T, config.r, config.A
    a_k = np.array(config.a_k)
    std_k = config.std_k
    hidden = config.dp_v2_hidden
    n_mc = config.dp_v2_mc_samples
    epochs = config.dp_v2_epochs
    lr = config.dp_v2_lr
    rng = np.random.default_rng(config.seed)

    action_grid = _build_action_grid(n, config.dp_v2_action_grid)
    n_act = len(action_grid)

    if verbose:
        print(f"[DP-v2] n={n}, T={T}, A={A}, states={config.dp_v2_state_samples}, "
              f"actions={n_act}, MC={n_mc}")

    input_dim = 1 + n + 1  # (wealth, cash, risky_1, ..., risky_n)
    value_nets = {}
    policy_nets = {}
    v_stats = {}

    if verbose:
        print(f"\n{'=' * 60}\nBackward Induction (DP-v2)\n{'=' * 60}")
    total_start = time.time()

    for t in reversed(range(T)):
        step_start = time.time()

        W, weights = _sample_states(config, t, rng)
        S = len(W)

        targets = np.full(S, -1e20)
        best_actions = np.zeros((S, n))

        for si in range(S):
            w = W[si]
            wts = weights[si]

            new_wts = _apply_actions(wts, action_grid, config)  # (A, n+1)

            # MC samples
            R = rng.normal(a_k, std_k, size=(n_mc, n))

            risky = new_wts[:, 1:]   # (A, n)
            cash = new_wts[:, 0]     # (A,)

            port_ret = cash[:, None] * r + risky @ R.T  # (A, n_mc)
            w_next = w * (1.0 + port_ret)               # (A, n_mc)

            if t == T - 1:
                vals = cara_utility(w_next, A)  # (A, n_mc)
            else:
                denom = 1.0 + port_ret
                new_cash_next = cash[:, None] * (1.0 + r) / denom   # (A, n_mc)
                new_risky_next = (risky[:, None, :] *
                                  (1.0 + R[None, :, :])) / denom[:, :, None]

                w_clip = np.clip(w_next, 0.001, 100.0)
                c_clip = np.clip(new_cash_next, 0.0, 1.0)
                r_clip = np.clip(new_risky_next, 0.0, 1.0)

                flat_w = w_clip.reshape(-1, 1)
                flat_c = c_clip.reshape(-1, 1)
                flat_r = r_clip.reshape(-1, n)
                nn_input = np.concatenate([flat_w, flat_c, flat_r], axis=1)

                with torch.no_grad():
                    inp_t = torch.tensor(nn_input, dtype=torch.float32)
                    raw = value_nets[t + 1](inp_t).numpy()
                    vals_flat = raw * v_stats[t + 1][1] + v_stats[t + 1][0]
                vals = vals_flat.reshape(n_act, n_mc)

                bankrupt = w_next <= 0.001
                if np.any(bankrupt):
                    vals[bankrupt] = cara_utility(0.001, A)

            ev = vals.mean(axis=1)  # (A,)
            best = np.argmax(ev)
            targets[si] = ev[best]
            best_actions[si] = action_grid[best]

        # Train value net
        valid = targets > -1e19
        n_valid = valid.sum()

        if n_valid < 10:
            if verbose:
                print(f"  t={t}: WARNING only {n_valid} valid states")
            vnet = ValueNet(input_dim, hidden)
            pnet = PolicyNet(input_dim, n, hidden)
            value_nets[t] = vnet
            policy_nets[t] = pnet
            v_stats[t] = (0.0, 1.0)
            continue

        X_v = np.column_stack([W[valid], weights[valid]])
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
        X_p = np.column_stack([W[valid], weights[valid]])
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
            print(f"  t={t}: {n_valid}/{S} valid, "
                  f"V=[{y_v.min():.3f}, {y_v.max():.3f}], "
                  f"{time.time() - step_start:.1f}s")

    if verbose:
        print(f"Total: {time.time() - total_start:.1f}s")

    result = {
        "value_nets": value_nets, "policy_nets": policy_nets,
        "v_stats": v_stats, "action_grid": action_grid,
        "config": {"n": n, "T": T},
    }
    save_cache(result, config, "dp_v2", verbose=verbose)
    return result


def get_policy_fn(solver_state: dict):
    """Create policy function: (obs, env) -> action in [-1,1]^n."""
    policy_nets = solver_state["policy_nets"]
    n = solver_state["config"]["n"]

    def policy_fn(obs, env):
        t = min(int(round(obs[0] * env.T)), env.T - 1)
        W = float(obs[1])
        p_all = obs[2:].astype(np.float64)  # [cash, risky_1, ..., risky_n]

        if t not in policy_nets:
            return np.zeros(env.n, dtype=np.float32)

        inp = torch.tensor(
            np.concatenate([[W], p_all]).reshape(1, -1),
            dtype=torch.float32,
        )
        with torch.no_grad():
            action = policy_nets[t](inp).numpy()[0]
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    return policy_fn
