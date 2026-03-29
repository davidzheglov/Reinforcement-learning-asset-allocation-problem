"""
adp_mc_solver.py -- Approximate DP with Monte Carlo expectations.

Method:
    Backward induction t = T-1 ... 0
    V_{t+1} approximated by a neural network (ValueNet)
    Expectation: Monte Carlo sampling of asset returns
    Action space: feasible deltas (pre-filtered by turnover)
    Policy: best discrete action stored, NN policy trained on labels

Training states sampled by rolling forward from p_init with random actions.
"""

from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn as nn

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


def solve(config: ScenarioConfig, verbose: bool = True) -> dict:
    """Solve via ADP backward induction with MC expectations."""
    cached = load_cache(config, "adp_mc", verbose=verbose)
    if cached is not None:
        return cached

    n, T, r, A = config.n, config.T, config.r, config.A
    a_k = np.array(config.a_k)
    std_k = config.std_k
    hidden = config.adp_hidden
    n_train = config.adp_n_train
    n_mc = config.adp_mc_samples
    epochs = config.adp_epochs
    lr = config.adp_lr
    rng = np.random.default_rng(config.seed)

    actions = build_feasible_actions(config, step=config.adp_action_step)
    n_act = len(actions)

    if verbose:
        print(f"[ADP-MC] n={n}, T={T}, A={A}, states={n_train}, "
              f"actions={n_act}, MC={n_mc}")

    input_dim = 1 + n  # (wealth, p_1, ..., p_n)
    value_nets = {}
    policy_nets = {}
    v_stats = {}

    if verbose:
        print(f"\n{'=' * 60}\nBackward Induction (ADP-MC)\n{'=' * 60}")
    total_start = time.time()

    for t in reversed(range(T)):
        step_start = time.time()

        # Sample training states at time t
        W, p_risky = sample_reachable_states(config, t, actions, rng, n_train)
        S = len(W)

        # Compute Bellman targets
        p_new, new_cash, feasible = check_feasibility(p_risky, actions, config)
        # p_new: (S, A, n), new_cash: (S, A), feasible: (S, A)

        # MC return samples: (n_mc, n)
        R_samples = rng.normal(a_k, std_k, size=(n_mc, n))

        # Compute expected values in chunks to manage memory
        # Memory per chunk: cs * A * n_mc * 8 bytes
        mem_budget = 200_000_000  # 200 MB
        per_state = n_act * n_mc * 8 * 3  # rough estimate
        chunk_size = max(1, mem_budget // per_state)

        targets = np.full(S, -1e20)
        best_actions = np.zeros((S, n))

        for s_start in range(0, S, chunk_size):
            s_end = min(s_start + chunk_size, S)
            cs = s_end - s_start

            p_new_c = p_new[s_start:s_end]       # (cs, A, n)
            nc_c = new_cash[s_start:s_end]         # (cs, A)
            feas_c = feasible[s_start:s_end]       # (cs, A)
            W_c = W[s_start:s_end]                 # (cs,)

            # Portfolio return: (cs, A, n_mc)
            port_ret = (nc_c[:, :, None] * r +
                        np.einsum('san,qn->saq', p_new_c, R_samples))
            w_next = W_c[:, None, None] * (1.0 + port_ret)  # (cs, A, n_mc)

            if t == T - 1:
                vals = cara_utility(w_next, A)  # (cs, A, n_mc)
            else:
                # Compute next proportions
                denom = 1.0 + port_ret  # (cs, A, n_mc)
                p_next = (p_new_c[:, :, None, :] *
                          (1.0 + R_samples[None, None, :, :])) / denom[:, :, :, None]
                # p_next: (cs, A, n_mc, n)

                w_clip = np.clip(w_next, 0.001, 100.0)
                p_clip = np.clip(p_next, 0.0, 1.0)

                # Build NN input: (cs * A * n_mc, 1+n)
                flat_w = w_clip.reshape(-1, 1)
                flat_p = p_clip.reshape(-1, n)
                nn_input = np.concatenate([flat_w, flat_p], axis=1)

                with torch.no_grad():
                    inp_t = torch.tensor(nn_input, dtype=torch.float32)
                    raw = value_nets[t + 1](inp_t).numpy()
                    vals_flat = raw * v_stats[t + 1][1] + v_stats[t + 1][0]
                vals = vals_flat.reshape(cs, n_act, n_mc)

                # Handle bankruptcy
                bankrupt = w_next <= 0.001
                if np.any(bankrupt):
                    vals[bankrupt] = cara_utility(0.001, A)

            # Expected value over MC samples: (cs, A)
            ev = vals.mean(axis=2)

            # Mask infeasible
            ev[~feas_c] = -1e20

            best_idx = np.argmax(ev, axis=1)  # (cs,)
            targets[s_start:s_end] = ev[np.arange(cs), best_idx]
            best_actions[s_start:s_end] = actions[best_idx]

        # Train value net
        valid = targets > -1e19
        if valid.sum() < 10:
            if verbose:
                print(f"  t={t}: WARNING only {valid.sum()} valid states")
            # Create dummy nets
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
    save_cache(result, config, "adp_mc", verbose=verbose)
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
