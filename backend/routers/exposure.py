from fastapi import APIRouter

from models.schemas import TradeInput, ExposureResponse
from simulation.gbm import generate_gbm_paths
from simulation.heston import generate_heston_paths
from simulation.merton import generate_merton_paths
from simulation.vasicek import generate_vasicek_paths
from simulation.cir import generate_cir_paths
from simulation.exposure import compute_mtm, compute_exposure_profile, compute_ene
from simulation.irs_exposure import compute_irs_mtm
from simulation.cva import compute_epe, compute_cva, compute_dva

router = APIRouter(prefix="/api")


@router.post("/exposure", response_model=ExposureResponse)
def calculate_exposure(trade: TradeInput) -> ExposureResponse:
    """
    Run Monte Carlo simulation and return the exposure profile + CVA metrics.
    Dispatches first on product (fx_forward | irs), then on sim_model.
    """
    # ── Step 1: Generate paths ──────────────────────────────────────────────
    if trade.product == "fx_forward":
        if trade.sim_model == "gbm":
            time_grid, paths = generate_gbm_paths(
                S0=trade.spot_rate,
                r_d=trade.r_d,
                r_f=trade.r_f,
                sigma=trade.volatility,
                T=trade.maturity,
                num_steps=trade.num_steps,
                num_paths=trade.num_paths,
                seed=42,
            )
        elif trade.sim_model == "heston":
            time_grid, paths = generate_heston_paths(
                S0=trade.spot_rate,
                r_d=trade.r_d,
                r_f=trade.r_f,
                v0=trade.heston_v0,
                kappa=trade.heston_kappa,
                theta=trade.heston_theta,
                xi=trade.heston_xi,
                rho=trade.heston_rho,
                T=trade.maturity,
                num_steps=trade.num_steps,
                num_paths=trade.num_paths,
                seed=42,
            )
        else:  # merton
            time_grid, paths = generate_merton_paths(
                S0=trade.spot_rate,
                r_d=trade.r_d,
                r_f=trade.r_f,
                sigma=trade.volatility,
                T=trade.maturity,
                num_steps=trade.num_steps,
                num_paths=trade.num_paths,
                jump_intensity=trade.merton_lambda,
                jump_mean=trade.merton_mu_j,
                jump_vol=trade.merton_sigma_j,
                seed=42,
            )
    else:  # irs
        if trade.sim_model == "vasicek":
            time_grid, paths = generate_vasicek_paths(
                r0=trade.irs_r0,
                kappa=trade.irs_kappa,
                theta=trade.irs_theta,
                sigma=trade.irs_vol,
                T=trade.maturity,
                num_steps=trade.num_steps,
                num_paths=trade.num_paths,
                seed=42,
            )
        else:  # cir
            time_grid, paths = generate_cir_paths(
                r0=trade.irs_r0,
                kappa=trade.irs_kappa,
                theta=trade.irs_theta,
                sigma=trade.irs_vol,
                T=trade.maturity,
                num_steps=trade.num_steps,
                num_paths=trade.num_paths,
                seed=42,
            )

    # ── Step 2: Compute MtM ──────────────────────────────────────────────────
    if trade.product == "fx_forward":
        mtm = compute_mtm(
            paths=paths,
            time_grid=time_grid,
            K=trade.strike_rate,
            r_d=trade.r_d,
            r_f=trade.r_f,
            notional=trade.notional,
            T=trade.maturity,
        )
    else:  # irs
        mtm = compute_irs_mtm(
            rate_paths=paths,
            time_grid=time_grid,
            K_fixed=trade.irs_fixed_rate,
            notional=trade.notional,
            T=trade.maturity,
            payment_freq=trade.irs_payment_freq,
            kappa=trade.irs_kappa,
            theta=trade.irs_theta,
            sigma=trade.irs_vol,
            model=trade.sim_model,
        )

    # Receiver IRS: pay fixed, receive floating → MtM is negated
    if trade.product == "irs" and trade.irs_direction == "receiver":
        mtm = -mtm

    # ── Steps 3–5: Exposure metrics ──────────────────────────────────────────
    ee, pfe = compute_exposure_profile(mtm, trade.pfe_confidence)
    ene  = compute_ene(mtm)
    epe  = compute_epe(ee, time_grid)
    cva  = compute_cva(ee,  time_grid, trade.cds_spread_bps,     trade.recovery_rate)
    dva  = compute_dva(ene, time_grid, trade.own_cds_spread_bps,  trade.own_recovery_rate)
    bcva = cva - dva

    return ExposureResponse(
        time_grid=time_grid.tolist(),
        ee=ee.tolist(),
        pfe=pfe.tolist(),
        ene=ene.tolist(),
        epe=epe,
        cva=cva,
        dva=dva,
        bcva=bcva,
        currency_pair=trade.currency_pair,
        notional=trade.notional,
        maturity=trade.maturity,
        pfe_confidence=trade.pfe_confidence,
        sim_model=trade.sim_model,
        product=trade.product,
    )
