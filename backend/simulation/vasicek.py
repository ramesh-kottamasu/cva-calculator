"""
Vasicek (1977) short-rate model — exact discretization.

    dr = κ(θ − r) dt + σ dW

Because this is an Ornstein-Uhlenbeck process, the transition distribution is
Gaussian with a closed-form mean and variance (no discretisation error):

    r(t+dt) | r(t)  ~  N( μ_dt,  σ_dt² )

    μ_dt  = r(t)·exp(−κ·dt) + θ·(1 − exp(−κ·dt))
    σ_dt² = σ²·(1 − exp(−2κ·dt)) / (2κ)

Antithetic variates applied to Z.
Returns shape (num_steps+1, num_paths) — matching FX path convention.
"""
import numpy as np


def generate_vasicek_paths(
    r0: float,
    kappa: float,
    theta: float,
    sigma: float,
    T: float,
    num_steps: int,
    num_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if num_paths % 2 != 0:
        num_paths += 1
    half = num_paths // 2

    dt = T / num_steps
    time_grid = np.linspace(0.0, T, num_steps + 1)

    e_kdt  = np.exp(-kappa * dt)
    e_2kdt = np.exp(-2.0 * kappa * dt)
    # Conditional mean = r(t)*e_kdt + mean_dt_const
    mean_const = theta * (1.0 - e_kdt)
    # Conditional std dev
    cond_std = sigma * np.sqrt((1.0 - e_2kdt) / (2.0 * kappa))

    # Pre-generate all randoms upfront
    Z_half = rng.standard_normal((num_steps, half))
    Z_full = np.concatenate([Z_half, -Z_half], axis=1)  # antithetic

    paths = np.empty((num_steps + 1, num_paths))
    paths[0] = r0
    for i in range(num_steps):
        paths[i + 1] = paths[i] * e_kdt + mean_const + cond_std * Z_full[i]

    return time_grid, paths
