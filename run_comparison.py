"""
run_comparison.py -- Compare multiple solvers on one or more scenarios.

Usage:
    python -m asset_allocation.run_comparison --scenario long_only_attractive
    python -m asset_allocation.run_comparison --scenario colab_demo --only hold heuristic tabular adp_mc
    python -m asset_allocation.run_comparison --all-scenarios --only hold heuristic tabular
    python -m asset_allocation.run_comparison --scenario long_only_attractive --plot

Solvers: hold, heuristic, tabular, adp_mc, adp_hermite, dp_v2, ppo, a2c
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path

from asset_allocation.config import SCENARIOS, ScenarioConfig
from asset_allocation.environment import PortfolioEnv
from asset_allocation.evaluation import (
    evaluate_policy, format_result, generate_shared_returns,
)
from asset_allocation.run_single import _get_solver_and_policy, SOLVER_NAMES

REPORTS_DIR = Path(__file__).parent / "outputs" / "reports"


def run_scenario(config: ScenarioConfig, solvers: list, n_episodes: int = None,
                 verbose: bool = True, plot: bool = False) -> dict:
    """Run all specified solvers on a scenario, return results dict."""
    n_ep = n_episodes or config.n_eval_episodes
    shared = generate_shared_returns(config)

    print(f"\n{'=' * 60}")
    print(f"Scenario: {config.summary()}")
    print(f"Solvers:  {', '.join(solvers)}")
    print(f"Episodes: {n_ep}")
    print(f"{'=' * 60}")

    results = {}
    for solver_name in solvers:
        print(f"\n--- {solver_name} ---")
        try:
            _, policy_fn = _get_solver_and_policy(solver_name, config, verbose=verbose)
            env = PortfolioEnv(config)
            res = evaluate_policy(env, policy_fn, n_episodes=n_ep,
                                  seed=config.seed, shared_returns=shared)
            results[solver_name] = res
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Summary table
    print(f"\n{'=' * 60}")
    print(f"Results: {config.name}")
    print(f"{'=' * 60}")
    for name, res in results.items():
        print(format_result(res, name))

    if plot and results:
        from asset_allocation.plotting import plot_all
        plot_all(results, config.name, config=config)

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare solvers")
    parser.add_argument("--scenario", "-s", default=None,
                        help="Scenario name")
    parser.add_argument("--all-scenarios", action="store_true",
                        help="Run on all predefined scenarios")
    parser.add_argument("--only", nargs="+", default=None,
                        help=f"Solvers to run. Default: all. Available: {', '.join(SOLVER_NAMES)}")
    parser.add_argument("--episodes", "-e", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--save-report", action="store_true",
                        help="Save JSON report to outputs/reports/")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    solvers = args.only or SOLVER_NAMES

    if args.all_scenarios:
        scenario_names = list(SCENARIOS.keys())
    elif args.scenario:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario '{args.scenario}'.")
            print(f"Available: {', '.join(sorted(SCENARIOS.keys()))}")
            return
        scenario_names = [args.scenario]
    else:
        print("Specify --scenario or --all-scenarios")
        return

    all_results = {}
    for sname in scenario_names:
        config = SCENARIOS[sname]
        if args.no_cache:
            config.cache_enabled = False

        results = run_scenario(
            config, solvers,
            n_episodes=args.episodes,
            verbose=not args.quiet,
            plot=args.plot,
        )
        all_results[sname] = {
            name: {
                "mean_utility": r["mean_utility"],
                "std_utility": r["std_utility"],
                "mean_wealth": r["mean_wealth"],
                "std_wealth": r["std_wealth"],
            }
            for name, r in results.items()
        }

    if args.save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = REPORTS_DIR / "comparison_report.json"
        with open(str(fname), "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nReport saved: {fname}")


if __name__ == "__main__":
    main()
