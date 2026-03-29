"""
asset_allocation -- Finite-horizon multi-asset portfolio allocation solvers.

Solvers:
    tabular_solver      Exact DP on discretized grids (Gauss-Hermite quadrature)
    adp_mc_solver       Approximate DP with Monte Carlo expectations
    adp_hermite_solver  Approximate DP with Gauss-Hermite expectations
    dp_v2_solver        Approximate DP v2 (MC + normalized action grid)
    ppo_solver          Proximal Policy Optimization (model-free RL)
    a2c_solver          Advantage Actor-Critic (model-free RL)

Framework:
    config              Scenario definitions and parameters
    environment         Gymnasium environment
    baselines           Hold and heuristic baseline policies
    evaluation          Unified rollout and comparison utilities
    plotting            Comparison plots, portfolio evolution, economic intuition
    caching             Disk caching for trained models
    utils               Shared helpers (CARA utility, action grids, executed deltas)

CLI:
    run_single          Run one solver on one scenario
    run_comparison      Compare multiple solvers across scenarios
    clean               Remove cached / generated outputs
"""

from asset_allocation.config import ScenarioConfig, SCENARIOS
