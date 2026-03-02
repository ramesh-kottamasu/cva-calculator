"""
Merton (1976) jump-diffusion model.

    dS/S = (r_d - r_f - λk̄) dt + σ dW + (e^J − 1) dN

where:
    N(t) ~ Poisson(λ)                     jump process
    J    ~ N(μⱼ, σⱼ²)                     log-jump size per jump
    k̄   = E[e^J] − 1 = exp(μⱼ + σⱼ²/2) − 1   risk-neutral drift correction

The λk̄ term removes the expected jump contribution from the drift so that
the discounted FX rate remains a martingale under the risk-neutral measure.

Compound Poisson sum: given N jumps in interval dt, the total log-jump is:
    Σ Jᵢ ~ N(N·μⱼ, N·σⱼ²)       (sum of i.i.d. normals)

This is exact (not an approximation) and fully vectorised via numpy.

Antithetic variates applied to the Brownian component only.  The Poisson
jump counts are left independent across the antithetic pair — this is the
standard practice because sign-flipping discrete Poisson draws would
distort the jump distribution.
"""
import numpy as np


def generate_merton_paths(
    S0: float,
    r_d: float,
    r_f: float,
    sigma: float,
    T: float,
    num_steps: int,
    num_paths: int,
    jump_intensity: float,
    jump_mean: float,
    jump_vol: float,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate Merton jump-diffusion FX rate paths.

    Args:
        S0:             Initial spot rate
        r_d, r_f:       Domestic / foreign risk-free rates
        sigma:          Continuous diffusion volatility σ
        T:              Maturity in years
        num_steps:      Number of time steps
        num_paths:      Number of Monte Carlo paths (forced even for antithetics)
        jump_intensity: λ — average number of jumps per year
        jump_mean:      μⱼ — mean log-jump size (log-return per jump)
        jump_vol:       σⱼ — std dev of log-jump size

    Returns:
        time_grid: shape (num_steps+1,)
        paths:     shape (num_steps+1, num_paths)
    """
    rng = np.random.default_rng(seed)

    if num_paths % 2 != 0:
        num_paths += 1
    half = num_paths // 2

    dt = T / num_steps
    time_grid = np.linspace(0.0, T, num_steps + 1)

    # ── Risk-neutral drift correction ──────────────────────────────────────
    k_bar = np.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0   # E[e^J] - 1
    drift     = (r_d - r_f - 0.5 * sigma ** 2 - jump_intensity * k_bar) * dt
    diffusion = sigma * np.sqrt(dt)

    # ── Continuous Brownian component (antithetic) ─────────────────────────
    Z_half = rng.standard_normal((num_steps, half))
    Z_full = np.concatenate([Z_half, -Z_half], axis=1)    # (num_steps, num_paths)

    # ── Jump component: compound Poisson ──────────────────────────────────
    lam_dt  = jump_intensity * dt
    n_jumps = rng.poisson(lam_dt, size=(num_steps, num_paths)).astype(float)

    # Conditional on N jumps: total log-jump ~ N(N·μⱼ, N·σⱼ²)
    # When N = 0 the result is identically 0 (handled by np.where)
    Z_jump    = rng.standard_normal((num_steps, num_paths))
    total_jump = np.where(
        n_jumps > 0,
        n_jumps * jump_mean + np.sqrt(n_jumps) * jump_vol * Z_jump,
        0.0,
    )

    # ── Build log-price paths ──────────────────────────────────────────────
    log_returns = drift + diffusion * Z_full + total_jump
    log_prices  = np.vstack([
        np.zeros((1, num_paths)),
        np.cumsum(log_returns, axis=0),
    ])

    return time_grid, S0 * np.exp(log_prices)
