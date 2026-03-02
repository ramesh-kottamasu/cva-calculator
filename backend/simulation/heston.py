"""
Heston (1993) stochastic volatility model.

    dS = (r_d - r_f) S dt + sqrt(v) S dW_S
    dv = κ(θ - v) dt + ξ sqrt(v) dW_v
    Corr(dW_S, dW_v) = ρ

Discretisation: Euler-Maruyama with full-truncation scheme.
Full truncation: use max(v, 0) in the drift and diffusion of v at each step.
This is numerically stable and prevents negative variance from compounding,
at the cost of a small discretisation bias (negligible for fine time grids).

Antithetic variates applied to both Brownian drivers jointly (−Z1, −Z_indep),
which preserves the correlation structure since:
    Z_v = ρ·Z1 + √(1-ρ²)·Z_indep  →  −Z_v = ρ·(−Z1) + √(1-ρ²)·(−Z_indep)
"""
import numpy as np


def generate_heston_paths(
    S0: float,
    r_d: float,
    r_f: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    T: float,
    num_steps: int,
    num_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Heston FX rate paths via Euler-Maruyama (full truncation).

    Returns:
        time_grid: shape (num_steps+1,)
        paths:     shape (num_steps+1, num_paths)  — FX spot rates
    """
    rng = np.random.default_rng(seed)

    if num_paths % 2 != 0:
        num_paths += 1
    half = num_paths // 2

    dt = T / num_steps
    sqrt_dt = np.sqrt(dt)
    rho_comp = np.sqrt(1.0 - rho ** 2)  # √(1 - ρ²) for Cholesky decomposition

    time_grid = np.linspace(0.0, T, num_steps + 1)

    # Pre-generate all random numbers upfront: (num_steps, 2, half)
    # axis-1 dimension 0 = Z1 (spot driver), dimension 1 = Z_indep (independent vol driver)
    Z_raw = rng.standard_normal((num_steps, 2, half))
    # Antithetic: concatenate with negatives → shape (num_steps, 2, num_paths)
    Z_all = np.concatenate([Z_raw, -Z_raw], axis=2)

    Z1    = Z_all[:, 0, :]                       # spot Brownian increments
    Z_ind = Z_all[:, 1, :]                       # independent component
    Z_v   = rho * Z1 + rho_comp * Z_ind          # correlated vol Brownian

    # Initialise path arrays using log-spot for numerical stability
    log_S = np.zeros((num_steps + 1, num_paths))
    log_S[0] = np.log(S0)
    v = np.full(num_paths, v0, dtype=float)

    for i in range(num_steps):
        v_pos   = np.maximum(v, 0.0)    # full truncation floor
        sqrt_v  = np.sqrt(v_pos)

        # Log-Euler update for spot (avoids negative prices)
        log_S[i + 1] = (
            log_S[i]
            + (r_d - r_f - 0.5 * v_pos) * dt
            + sqrt_v * sqrt_dt * Z1[i]
        )

        # Euler update for variance (full-truncation: drift/diffusion use v_pos)
        v = v + kappa * (theta - v_pos) * dt + xi * sqrt_v * sqrt_dt * Z_v[i]

    return time_grid, np.exp(log_S)
