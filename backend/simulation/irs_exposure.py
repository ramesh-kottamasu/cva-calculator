"""
Payer IRS MtM using affine term-structure bond prices (Vasicek / CIR).

At each Monte Carlo time step t with short rate r(t), the payer IRS value is:

    MtM(t, r) = N × [(1 − P(t, T)) − K_fixed × annuity(t, r)]

where the annuity sums over all future payment dates T_i > t:

    annuity(t, r) = Σ_{T_i > t} δ × P(t, T_i)

Bond prices via affine term structure:  P(τ) = exp(A(τ) − B(τ)·r)

Vasicek coefficients:
    B(τ) = (1 − exp(−κτ)) / κ
    A(τ) = (B − τ)(κ²θ − σ²/2) / κ² − σ²B² / (4κ)

CIR coefficients  (γ = √(κ² + 2σ²)):
    denom = (γ + κ)(exp(γτ) − 1) + 2γ
    B(τ)  = 2(exp(γτ) − 1) / denom
    A(τ)  = (2κθ/σ²) · ln(2γ · exp((γ+κ)τ/2) / denom)
"""
import numpy as np


# ── Affine coefficients ───────────────────────────────────────────────────────

def _vasicek_AB(
    tau: np.ndarray, kappa: float, theta: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    B = (1.0 - np.exp(-kappa * tau)) / kappa
    A = ((B - tau) * (kappa ** 2 * theta - 0.5 * sigma ** 2) / kappa ** 2
         - sigma ** 2 * B ** 2 / (4.0 * kappa))
    return A, B


def _cir_AB(
    tau: np.ndarray, kappa: float, theta: float, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    gamma = np.sqrt(kappa ** 2 + 2.0 * sigma ** 2)
    exp_gt = np.exp(gamma * tau)
    denom = (gamma + kappa) * (exp_gt - 1.0) + 2.0 * gamma
    B = 2.0 * (exp_gt - 1.0) / denom
    A = (2.0 * kappa * theta / sigma ** 2) * np.log(
        2.0 * gamma * np.exp(0.5 * (gamma + kappa) * tau) / denom
    )
    return A, B


def _get_AB(model: str, tau: np.ndarray, kappa: float, theta: float, sigma: float):
    if model == "vasicek":
        return _vasicek_AB(tau, kappa, theta, sigma)
    return _cir_AB(tau, kappa, theta, sigma)


# ── Main function ─────────────────────────────────────────────────────────────

def compute_irs_mtm(
    rate_paths: np.ndarray,  # (num_steps+1, num_paths)
    time_grid: np.ndarray,   # (num_steps+1,)
    K_fixed: float,
    notional: float,
    T: float,
    payment_freq: int,       # payments per year (2 = semi-annual)
    kappa: float,
    theta: float,
    sigma: float,
    model: str,              # 'vasicek' | 'cir'
) -> np.ndarray:             # (num_steps+1, num_paths)
    """
    Payer IRS MtM at each (time-step, path) using affine bond prices.
    """
    n_times, num_paths = rate_paths.shape
    mtm = np.zeros((n_times, num_paths))

    # Fixed payment schedule: T_1, T_2, …, T_n (excluding 0, including T)
    n_pay = int(round(T * payment_freq))
    pay_dates = np.array([k / payment_freq for k in range(1, n_pay + 1)])
    delta = 1.0 / payment_freq

    for j in range(n_times):
        t = time_grid[j]
        tau_T = T - t
        if tau_T < 1e-10:
            mtm[j] = 0.0
            continue

        r = rate_paths[j]  # (num_paths,)

        # Bond price to swap maturity: P(t, T)
        A_T, B_T = _get_AB(model, np.array([tau_T]), kappa, theta, sigma)
        P_T = np.exp(A_T[0] - B_T[0] * r)   # (num_paths,)

        # Remaining payment dates (strictly after t)
        rem = pay_dates[pay_dates > t + 1e-10]
        if len(rem) == 0:
            mtm[j] = 0.0
            continue

        tau_i = rem - t  # (k,)
        A_i, B_i = _get_AB(model, tau_i, kappa, theta, sigma)
        # P_i[k, p] = exp(A_i[k] - B_i[k] * r[p])
        P_i = np.exp(A_i[:, None] - B_i[:, None] * r[None, :])  # (k, num_paths)

        annuity = delta * P_i.sum(axis=0)   # (num_paths,)
        mtm[j] = notional * ((1.0 - P_T) - K_fixed * annuity)

    return mtm
