"""
Cox-Ingersoll-Ross (1985) short-rate model — full-truncation Euler.

    dr = κ(θ − r) dt + σ √r dW

Full-truncation scheme (Lord et al. 2010):

    r̃(t+dt) = r_pos(t) + κ·(θ − r_pos(t))·dt + σ·√r_pos(t)·√dt·Z
    r(t+dt)  = max(r̃(t+dt), 0)

where r_pos(t) = max(r(t), 0).

Antithetic variates applied to Z.
Returns shape (num_steps+1, num_paths) — matching Vasicek convention.
"""
import numpy as np


def generate_cir_paths(
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
    sqrt_dt = np.sqrt(dt)
    time_grid = np.linspace(0.0, T, num_steps + 1)

    # Pre-generate all randoms upfront
    Z_half = rng.standard_normal((num_steps, half))
    Z_full = np.concatenate([Z_half, -Z_half], axis=1)  # antithetic

    paths = np.empty((num_steps + 1, num_paths))
    paths[0] = r0
    for i in range(num_steps):
        r_pos = np.maximum(paths[i], 0.0)
        drift = kappa * (theta - r_pos) * dt
        diffusion = sigma * np.sqrt(r_pos) * sqrt_dt * Z_full[i]
        paths[i + 1] = np.maximum(paths[i] + drift + diffusion, 0.0)

    return time_grid, paths
