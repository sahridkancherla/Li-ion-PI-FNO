"""Physics residual for SPM PINO: PDE + BCs + IC, non-dimensionalised.

Includes temporal-causality weighting applied to the PDE residual. This is a training
-causality mechanism, not an architectural one: it does not restrict what information
reaches the model's output (the FNO's FFT still mixes all timesteps, by
construction - this cannot be changed without abandoning the FNO
architecture). Instead it reweights how much each timestep's PDE residual
contributes to the loss, so timesteps are not optimised until earlier
timesteps already have small residual. This is compatible with FNO's global
Fourier structure since it changes WHEN the optimizer focuses on each
timestep's error, not what the architecture can see.
"""
import torch
from utils import reshape_dx

F = 96485.3329  # Faraday's constant in C/mol


def central_diff(f, dx, dim):
    df = torch.zeros_like(f)

    def slc(start, stop, step=1):
        idx = [slice(None)] * f.ndim
        idx[dim] = slice(start, stop, step)
        return tuple(idx)

    N = f.shape[dim]
    dx = reshape_dx(dx, f, dim)

    df[slc(1, N - 1)] = (f[slc(2, N)] - f[slc(0, N - 2)]) / (2 * dx)
    df[slc(0, 1)] = (-3 * f[slc(0, 1)] + 4 * f[slc(1, 2)] - f[slc(2, 3)]) / (2 * dx)
    df[slc(N - 1, N)] = (3 * f[slc(N - 1, N)] - 4 * f[slc(N - 2, N - 1)] + f[slc(N - 3, N - 2)]) / (2 * dx)

    return df


def second_diff(f, dx, dim):
    d2f = torch.zeros_like(f)

    def slc(start, stop, step=1):
        idx = [slice(None)] * f.ndim
        idx[dim] = slice(start, stop, step)
        return tuple(idx)

    N = f.shape[dim]
    dx = reshape_dx(dx, f, dim)

    d2f[slc(1, N - 1)] = (f[slc(2, N)] + f[slc(0, N - 2)] - 2 * f[slc(1, N - 1)]) / dx**2
    return d2f


def pde_residual_per_timestep(c, r_phys, D_s, dr, dt, c_scale, R):
    """Same physics as pde_residual, but returns the squared residual
    averaged over batch and radius while KEEPING the time axis, so that
    temporal-causality weighting can be applied before the final reduction.

    Returns: [n_t] tensor (note: the r=0 row is excluded, same as before).
    """
    dc_dt = central_diff(c, dt, dim=-1)
    dc_dr = central_diff(c, dr, dim=-2)

    r = r_phys.view(1, -1, 1)
    r2_dc_dr = r ** 2 * dc_dr
    d_dr_r2_dc_dr = central_diff(r2_dc_dr, dr, dim=-2)

    residual = dc_dt - (D_s / (r ** 2 + 1e-12)) * d_dr_r2_dc_dr

    t_scale = R ** 2 / D_s
    residual_scale = c_scale / t_scale
    residual_nd = residual / residual_scale

    # average over batch (dim 0) and radius (dim 1, excluding r=0), keep time (dim 2)
    return (residual_nd[:, 1:, :] ** 2).mean(dim=(0, 1))  # -> [n_t]


def pde_residual(c, r_phys, D_s, dr, dt, c_scale, R):
    """Original scalar PDE residual (no temporal weighting) -- kept for
    backward compatibility / use in bc and ic residuals' calling convention.
    """
    return pde_residual_per_timestep(c, r_phys, D_s, dr, dt, c_scale, R).mean()


def temporal_causal_weights(loss_per_timestep, gamma=1.0):
    """
    Training-causality weighting per Wang et al. 2022:
        w_i = exp(-gamma * sum_{k<i} L(t_k))

    Does NOT restrict what the model can see (the FNO architecture still
    mixes all timesteps via FFT) -- only controls how much each timestep's
    residual contributes to the loss the optimizer sees, so early-time
    residuals must shrink before later ones are emphasised.

    loss_per_timestep: [n_t] tensor, e.g. from pde_residual_per_timestep.
    gamma: scaling hyperparameter controlling how strongly later timesteps
           are suppressed while earlier ones remain unsatisfied. Larger
           gamma = stricter ordering, smaller gamma = closer to uniform
           weighting (gamma=0 recovers the unweighted loss).

    Returns: [n_t] tensor of weights in (0, 1], same shape as input.
    Weights are detached -- they modulate the loss landscape based on the
    CURRENT state of training, but are not themselves differentiated through.
    """
    cumulative = torch.cumsum(loss_per_timestep.detach(), dim=0)
    cumulative_shifted = torch.cat([
        torch.zeros(1, device=loss_per_timestep.device, dtype=loss_per_timestep.dtype),
        cumulative[:-1]
    ])
    weights = torch.exp(-gamma * cumulative_shifted)
    return weights


def weighted_pde_loss(c, r_phys, D_s, dr, dt, c_scale, R, gamma=1.0):
    """Convenience wrapper: computes per-timestep PDE residual, applies
    temporal-causality weighting, and returns the final weighted scalar loss.
    """
    loss_per_t = pde_residual_per_timestep(c, r_phys, D_s, dr, dt, c_scale, R)
    weights = temporal_causal_weights(loss_per_t, gamma=gamma)
    return (weights * loss_per_t).mean()


def bc_surface_residual(c, j_surf, D_s, dr, c_scale, R):
    dc_dr = central_diff(c, dr, dim=-2)
    dc_dr_surf = dc_dr[:, -1, :]

    residual = dc_dr_surf + j_surf / (F * D_s)

    grad_scale = c_scale / R
    residual_nd = residual / grad_scale

    return (residual_nd ** 2).mean()


def bc_centre_residual(c, dr, c_scale, R):
    dc_dr = central_diff(c, dr, dim=-2)
    dc_dr_centre = dc_dr[:, 0, :]

    grad_scale = c_scale / R
    residual_nd = dc_dr_centre / grad_scale

    return (residual_nd ** 2).mean()


def ic_residual(c, c0, c_scale):
    c_t0 = c[:, :, 0]
    residual_nd = (c_t0 - c0) / c_scale

    return (residual_nd ** 2).mean()


def physics_loss(c_p, c_n, j_surf_p, j_surf_n,
                 r_phys_p, r_phys_n, dr_p, dr_n,
                 D_s_p, D_s_n, c0_p, c0_n, dt,
                 c_scale_p, c_scale_n, R_p, R_n,
                 reg_weights=None, gamma=1.0, use_temporal_causality=True):
    """
    Combined, non-dimensionalised physics loss: PDE + BCs + IC.
    p, n refers to +ve, -ve electrodes, respectively.

    gamma: temporal-causality strength (see temporal_causal_weights). Only
           affects the PDE term -- BC/IC residuals are not time-sequential
           in the same sense (BC is evaluated at every t independently with
           no notion of "earlier" being a prerequisite; IC is only at t=0).
    use_temporal_causality: if False, falls back to plain unweighted PDE
           loss (equivalent to gamma=0), useful for direct ablation.
    """
    if reg_weights is None:
        reg_weights = {"pde": 1.0, "bc_surf": 1.0, "bc_centre": 1.0, "ic": 1.0}

    if use_temporal_causality:
        pde_loss = (weighted_pde_loss(c_p, r_phys_p, D_s_p, dr_p, dt, c_scale_p, R_p, gamma=gamma) +
                    weighted_pde_loss(c_n, r_phys_n, D_s_n, dr_n, dt, c_scale_n, R_n, gamma=gamma))
    else:
        pde_loss = (pde_residual(c_p, r_phys_p, D_s_p, dr_p, dt, c_scale_p, R_p) +
                    pde_residual(c_n, r_phys_n, D_s_n, dr_n, dt, c_scale_n, R_n))

    bc_centre_loss = (bc_centre_residual(c_p, dr_p, c_scale_p, R_p) +
                       bc_centre_residual(c_n, dr_n, c_scale_n, R_n))

    bc_surface_loss = (bc_surface_residual(c_p, j_surf_p, D_s_p, dr_p, c_scale_p, R_p) +
                       bc_surface_residual(c_n, j_surf_n, D_s_n, dr_n, c_scale_n, R_n))

    ic_loss = (ic_residual(c_p, c0_p, c_scale_p) + ic_residual(c_n, c0_n, c_scale_n))

    total_loss = (reg_weights["pde"] * pde_loss +
                  reg_weights["bc_surf"] * bc_surface_loss +
                  reg_weights["bc_centre"] * bc_centre_loss +
                  reg_weights["ic"] * ic_loss)

    return total_loss, {"pde_loss": pde_loss.item(), "bc_surface_loss": bc_surface_loss.item(),
                        "bc_centre_loss": bc_centre_loss.item(), "ic_loss": ic_loss.item()}