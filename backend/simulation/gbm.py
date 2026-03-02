"""
GBM path generation for FX rates using antithetic variates.

S(t+dt) = S(t) * exp((r_d - r_f - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
where Z ~ N(0,1)

Antithetic variates: generate Z for half the paths, use -Z for the other half.
Path matrix shape: (num_steps+1, num_paths)
"""
import numpy as np


def generate_gbm_paths(
    S0: float,
    r_d: float,
    r_f: float,
    sigma: float,
    T: float,
    num_steps: int,
    num_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate GBM FX rate paths with antithetic variates.

    Returns:
        time_grid: shape (num_steps+1,) — time points t0=0..T
        paths:     shape (num_steps+1, num_paths) — S(t) across paths
    """
    rng = np.random.default_rng(seed)

    # Force num_paths to be even for antithetic variates
    if num_paths % 2 != 0:
        num_paths += 1

    half = num_paths // 2
    dt = T / num_steps

    time_grid = np.linspace(0.0, T, num_steps + 1)

    # GBM drift and diffusion terms per step
    drift = (r_d - r_f - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)

    # Draw standard normals for half the paths: shape (num_steps, half)
    Z = rng.standard_normal((num_steps, half))
    # Antithetic: stack Z and -Z
    Z_full = np.concatenate([Z, -Z], axis=1)  # shape (num_steps, num_paths)

    # Compute log-returns per step, then cumulative sum to get log-price path
    log_returns = drift + diffusion * Z_full  # (num_steps, num_paths)

    # Build paths: row 0 = S0, rows 1..num_steps = S0 * exp(cumsum of log_returns)
    log_prices = np.vstack([
        np.zeros((1, num_paths)),
        np.cumsum(log_returns, axis=0),
    ])  # shape (num_steps+1, num_paths)

    paths = S0 * np.exp(log_prices)

    return time_grid, paths
