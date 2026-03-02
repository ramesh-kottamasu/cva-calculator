"""
CVA (Credit Valuation Adjustment) calculations.

EPE = (1/T) * integral_0^T EE(t) dt          [trapezoidal rule]
CVA = (1-R) * sum_i EE(t_i) * DeltaPD(t_i)
where:
    DeltaPD(t_i) = PD(t_i) - PD(t_{i-1})
    PD(t) = 1 - exp(-lambda * t)
    lambda = CDS_bps / 10000 / (1 - R)        [implied hazard rate]
"""
import numpy as np


def compute_hazard_rate(cds_spread_bps: float, recovery_rate: float) -> float:
    """
    Convert CDS spread (in bps) to constant hazard rate lambda.

    Under the standard CDS pricing approximation:
        lambda = spread / (1 - R)
    where spread is in decimal form.
    """
    spread_decimal = cds_spread_bps / 10_000.0
    return spread_decimal / (1.0 - recovery_rate)


def compute_epe(ee: np.ndarray, time_grid: np.ndarray) -> float:
    """
    Compute Expected Positive Exposure (EPE) via trapezoidal integration.

    EPE = (1/T) * integral_0^T EE(t) dt
    """
    T = time_grid[-1]
    return float(np.trapezoid(ee, time_grid) / T)


def compute_cva(
    ee: np.ndarray,
    time_grid: np.ndarray,
    cds_spread_bps: float,
    recovery_rate: float,
) -> float:
    """
    Compute CVA using the discrete hazard-rate formula.

    CVA = (1-R) * sum_i EE(t_i) * DeltaPD(t_i)
    """
    if cds_spread_bps == 0.0:
        return 0.0

    lam = compute_hazard_rate(cds_spread_bps, recovery_rate)

    # Marginal default probabilities: DeltaPD(t_i) = PD(t_i) - PD(t_{i-1})
    pd = 1.0 - np.exp(-lam * time_grid)  # shape (num_steps+1,)
    delta_pd = np.diff(pd, prepend=0.0)   # DeltaPD[0] = PD(t_0) - 0 = PD(0) = 0

    cva = float((1.0 - recovery_rate) * np.sum(ee * delta_pd))
    return cva


def compute_dva(
    ene: np.ndarray,
    time_grid: np.ndarray,
    own_cds_spread_bps: float,
    own_recovery_rate: float,
) -> float:
    """
    Compute DVA (Debt Valuation Adjustment) using the discrete hazard-rate formula.

    DVA = (1 - R_own) * sum_i ENE(t_i) * DeltaPD_own(t_i)

    where ENE(t) = E[max(-MtM(t), 0)] is the expected negative exposure
    (our liability to the counterparty), and PD_own is our own default probability.

    DVA is the benefit we receive from our own credit risk — the counterparty
    implicitly writes off part of our liability if we may default.
    BCVA = CVA - DVA is the bilateral adjustment.
    """
    if own_cds_spread_bps == 0.0:
        return 0.0

    lam_own = compute_hazard_rate(own_cds_spread_bps, own_recovery_rate)

    pd_own = 1.0 - np.exp(-lam_own * time_grid)
    delta_pd_own = np.diff(pd_own, prepend=0.0)

    dva = float((1.0 - own_recovery_rate) * np.sum(ene * delta_pd_own))
    return dva
