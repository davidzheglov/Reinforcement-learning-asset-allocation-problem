"""
run_single.py -- Run a single solver on a single scenario.

Usage:
    python -m asset_allocation.run_single --scenario long_only_attractive --solver tabular
    python -m asset_allocation.run_single --scenario colab_demo --solver adp_mc --no-cache
    python -m asset_allocation.run_single --scenario two_assets --solver ppo --plot

Solvers: hold, heuristic, tabular, adp_mc, adp_hermite, dp_v2, ppo, a2c
"""

from __future__ import annotations
import argparse
import numpy as np

from asset_allocation.config import SCENARIOS, ScenarioConfig
from asset_allocation.environment import PortfolioEnv
from asset_allocation.evaluation import evaluate_policy, format_result, generate_shared_returns


SOLVER_NAMES = ["hold", "heuristic", "tabular", "adp_mc", "adp_hermite", "dp_v2", "ppo", "a2c"]


def _get_solver_and_policy(solver_name: str, config: ScenarioConfig, verbose: bool = True):
    """Return (solver_state_or_None, policy_fn)."""
    if solver_name == "hold":
        from asset_allocation.baselines import hold_policy
        return None, hold_policy
    elif solver_name == "heuristic":
        from asset_allocation.baselines import heuristic_policy
        return None, heuristic_policy
    elif solver_name == "tabular":
        from asset_allocation.tabular_solver import solve, get_policy_fn
        state = solve(config, verbose=verbose)
        return state, get_policy_fn(state)
    elif solver_name == "adp_mc":
        from asset_allocation.adp_mc_solver import solve, get_policy_fn
        state = solve(config, verbose=verbose)
        return state, get_policy_fn(state)
    elif solver_name == "adp_hermite":
        from asset_allocation.adp_hermite_solver import solve, get_policy_fn
        state = solve(config, verbose=verbose)
        return state, get_policy_fn(state)
    elif solver_name == "dp_v2":
        from asset_allocation.dp_v2_solver import solve, get_policy_fn
        state = solve(config, verbose=verbose)
        return state, get_policy_fn(state)
    elif solver_name == "ppo":
        from asset_allocation.ppo_solver import solve, get_policy_fn
        state = solve(config, verbose=verbose)
        return state, get_policy_fn(state)
    elif solver_name == "a2c":
        from asset_allocation.a2c_solver import solve, get_policy_fn
        state = solve(config, verbose=verbose)
        return state, get_policy_fn(state)
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def main():
    parser = argparse.ArgumentParser(description="Run single solver on one scenario")
    parser.add_argument("--scenario", "-s", required=True,
                        help=f"Scenario name. Available: {', '.join(SCENARIOS.keys())}")
    parser.add_argument("--solver", required=True,
                        help=f"Solver name. Available: {', '.join(SOLVER_NAMES)}")
    parser.add_argument("--episodes", "-e", type=int, default=None,
                        help="Number of eval episodes (default: from config)")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    args = parser.parse_args()

    if args.scenario not in SCENARIOS:
        print(f"Unknown scenario '{args.scenario}'.")
        print(f"Available: {', '.join(sorted(SCENARIOS.keys()))}")
        return

    config = SCENARIOS[args.scenario]
    if args.no_cache:
        config.cache_enabled = False

    n_episodes = args.episodes or config.n_eval_episodes

    print(f"\n{'=' * 60}")
    print(f"Scenario: {config.summary()}")
    print(f"Solver:   {args.solver}")
    print(f"Episodes: {n_episodes}")
    print(f"{'=' * 60}\n")

    # Train / load
    _, policy_fn = _get_solver_and_policy(args.solver, config, verbose=args.verbose)

    # Evaluate
    env = PortfolioEnv(config)
    shared = generate_shared_returns(config)
    result = evaluate_policy(env, policy_fn, n_episodes=n_episodes,
                             seed=config.seed, shared_returns=shared)

    print(f"\n{format_result(result, args.solver)}")

    if args.plot:
        from asset_allocation.plotting import plot_all
        plot_all({args.solver: result}, config.name, config=config)


if __name__ == "__main__":
    main()
