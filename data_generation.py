"""Generate SPM model data."""
import numpy as np
import pybamm

# Config
n_r = 32    # radial points -- increased from 20 to test whether the physics
             # loss plateau seen at coarser resolution is a finite-difference
             # resolution limit rather than a training issue
n_t = 500   # temporal points
n_train, n_val, n_test = 4000, 500, 500 # 80-10-10 split
n_samples = n_train + n_val + n_test
max_steps = 5 # max piecewise segments per profile
seed = 69
np.random.seed(seed)

# PyBAMM Setup
model = pybamm.lithium_ion.SPM()
param_vals = pybamm.ParameterValues("Chen2020")
capacity = param_vals["Nominal cell capacity [A.h]"]

# Override the default particle mesh resolution (PyBaMM defaults to 20 points
# for both r_n and r_p) -- without this, changing n_r above does NOT actually
# change PyBaMM's internal discretisation, and the returned arrays would still
# be shape 20 regardless of what n_r says, causing a silent shape mismatch.
var_pts = model.default_var_pts.copy()
var_pts["r_n"] = n_r
var_pts["r_p"] = n_r

# Storage - all shapes are [n_samples, n_r, n_t]
# concentrations
c_sn_all = np.zeros((n_samples, n_r, n_t))
c_sp_all = np.zeros((n_samples, n_r, n_t))
# potentials, broadcast across r
phi_sn_all = np.zeros((n_samples, n_r, n_t))
phi_sp_all = np.zeros((n_samples, n_r, n_t))
# interfacial current densities, broadcast across r -- needed for the BC residual
j_n_all = np.zeros((n_samples, n_r, n_t))
j_p_all = np.zeros((n_samples, n_r, n_t))
# inputs broadcast across r
I_all = np.zeros((n_samples, n_r, n_t))
t_all = np.zeros((n_samples, n_r, n_t))
# raw duration per sample, needed to convert normalised t back to physical seconds
t_end_all = np.zeros((n_samples,))
# radial coordinate broadcast across t, same for all samples
r_norm = np.linspace(0, 1, n_r)  # normalised radial grid -- NOTE: renamed from `r`
                                   # to avoid being shadowed by the loop variable below


def make_experiment():
    """Randomly choose between several profile types for current diversity:
    - constant: single C-rate discharge to cutoff
    - piecewise: multi-segment discharge, randomised C-rate and duration per segment
    - charge_discharge: partial discharge, rest, partial charge -- exercises both
      current directions, which the original discharge-only profiles never did
    - rest_discharge: discharge, rest (zero current relaxation), then discharge to
      cutoff -- exercises pure diffusion relaxation with no applied current, useful
      since the PDE residual is most informative when transport dominates over kinetics
    """
    profile_type = np.random.choice(
        ["constant", "piecewise", "charge_discharge", "rest_discharge"],
        p=[0.2, 0.5, 0.15, 0.15]
    )

    if profile_type == "constant":
        c = np.random.uniform(0.1, 3.0)
        return pybamm.Experiment([f"Discharge at {c:.2f}C until 2.5V"])

    elif profile_type == "piecewise":
        n_steps = np.random.randint(2, max_steps + 1)
        steps = []
        for _ in range(n_steps):
            c = np.random.uniform(0.1, 3.0)  # widened to match constant range --
                                               # previously capped at 2.0C, so high-rate
                                               # segments never appeared inside a sequence
            duration = np.random.randint(30, 900)  # widened from (60,600) for more
                                                      # short-transient and long-steady coverage
            steps.append(f"Discharge at {c:.2f}C for {duration} seconds or until 2.5V")
        return pybamm.Experiment(steps)

    elif profile_type == "charge_discharge":
        c_dis = np.random.uniform(0.2, 2.0)
        c_chg = np.random.uniform(0.1, 1.0)
        dur1 = np.random.randint(60, 600)
        return pybamm.Experiment([
            f"Discharge at {c_dis:.2f}C for {dur1} seconds",
            "Rest for 300 seconds",
            f"Charge at {c_chg:.2f}C for {dur1 // 2} seconds or until 4.2V",
        ])

    else:  # rest_discharge
        c = np.random.uniform(0.2, 2.0)
        dur = np.random.randint(60, 400)
        return pybamm.Experiment([
            f"Discharge at {c:.2f}C for {dur} seconds",
            "Rest for 600 seconds",
            f"Discharge at {c:.2f}C until 2.5V",
        ])

# Data Generation Loop
successes = 0
attempts = 0

while successes < n_samples:
    attempts += 1
    try:
        exp = make_experiment()
        sim = pybamm.Simulation(model, parameter_values=param_vals, experiment=exp, var_pts=var_pts)
        soln = sim.solve()

        # raw time axis
        t_raw = soln["Time [s]"].entries
        t_end = t_raw[-1]

        # uniform time axis
        t_uniform = np.linspace(0, t_end, n_t)
        t_norm = t_uniform / t_end  # normalised

        # get 1D timeseries, interpolate and then broadcast to [n_r, n_t]
        I_raw = soln["Current [A]"].entries
        phi_sn_raw = soln["X-averaged negative electrode potential [V]"].entries
        phi_sp_raw = soln["X-averaged positive electrode potential [V]"].entries
        j_n_raw = soln["X-averaged negative electrode interfacial current density [A.m-2]"].entries
        j_p_raw = soln["X-averaged positive electrode interfacial current density [A.m-2]"].entries

        I_re = np.interp(t_uniform, t_raw, I_raw)
        phi_sn_re = np.interp(t_uniform, t_raw, phi_sn_raw)
        phi_sp_re = np.interp(t_uniform, t_raw, phi_sp_raw)
        j_n_re = np.interp(t_uniform, t_raw, j_n_raw)
        j_p_re = np.interp(t_uniform, t_raw, j_p_raw)

        # concentration fields, already 2D just resampling in time
        c_sn_raw = soln["X-averaged negative particle concentration [mol.m-3]"].entries  # [n_r, n_t_raw]
        c_sp_raw = soln["X-averaged positive particle concentration [mol.m-3]"].entries  # [n_r, n_t_raw]

        # renamed inner loop variable to `ir` so it can never shadow the
        # outer radial-coordinate array (this was a real bug previously:
        # `r` here clobbered the global `r = np.linspace(...)`)
        c_sn_re = np.array([np.interp(t_uniform, t_raw, c_sn_raw[ir, :]) for ir in range(n_r)])  # [n_r, n_t]
        c_sp_re = np.array([np.interp(t_uniform, t_raw, c_sp_raw[ir, :]) for ir in range(n_r)])  # [n_r, n_t]

        # broadcast 1D fields to [n_r, n_t]
        I_all[successes] = np.tile(I_re, (n_r, 1))
        t_all[successes] = np.tile(t_norm, (n_r, 1))
        phi_sn_all[successes] = np.tile(phi_sn_re, (n_r, 1))
        phi_sp_all[successes] = np.tile(phi_sp_re, (n_r, 1))
        j_n_all[successes] = np.tile(j_n_re, (n_r, 1))
        j_p_all[successes] = np.tile(j_p_re, (n_r, 1))
        c_sn_all[successes] = c_sn_re
        c_sp_all[successes] = c_sp_re
        t_end_all[successes] = t_end

        successes += 1
        if successes % 50 == 0:
            print(f"{successes} simulations complete (attempts: {attempts})")

    except Exception as e:
        print(f"Simulation failed on attempt {attempts}: {e}")
        continue

print(f"Data generation complete: {successes} successful simulations after {attempts} attempts.")

# split
idx = np.random.permutation(n_samples)
train_idx, val_idx, test_idx = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def split_and_save(arr, name, train_idx, val_idx, test_idx):
    return {
        f"{name}_train": arr[train_idx],
        f"{name}_val": arr[val_idx],
        f"{name}_test": arr[test_idx]
    }

save_dict = {}
for arr, name in [
    (I_all, "I"),
    (t_all, "t"),
    (c_sn_all, "c_sn"),
    (c_sp_all, "c_sp"),
    (phi_sn_all, "phi_sn"),
    (phi_sp_all, "phi_sp"),
    (j_n_all, "j_n"),
    (j_p_all, "j_p"),
    (t_end_all, "t_end"),   # new: per-sample raw duration, needed for physics loss dt
]:
    save_dict.update(split_and_save(arr, name, train_idx, val_idx, test_idx))

# also save radial coordinates (same for every sample, not split)
save_dict["r_norm"] = r_norm
np.savez("spm_data_v3.npz", **save_dict)
print("Data saved to spm_data_v3.npz")