"""Hyperparameter sweep for the SPM PINO using Optuna.

Sweeps: lr, weight_decay, hidden_channels, n_layers
Fixed:  gamma, lambda_phys, n_modes_r, n_modes_t, batch_size, n_r

Pruning: MedianPruner kills trials that are clearly underperforming the
median of all completed trials at the same epoch, after a warmup period.
This makes the sweep tractable for expensive models.
"""
import torch
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner

from data_loader import get_dataloaders
from neuralop.models import FNO
from spm_constants import get_spm_const
from physics_loss import physics_loss, pde_residual_per_timestep, temporal_causal_weights
from utils import model_input, physical_grid

# ── fixed config ──────────────────────────────────────────────────────────────
FILE_NAME = "spm_data_v3.npz"
N_BATCH = 32
N_EPOCHS = 50          # max epochs per trial (pruner will kill bad ones early)
WARMUP_EPOCHS = 10      # pruner doesn't act before this many epochs
N_TRIALS = 50           # total number of hyperparameter combinations to try
LAMBDA_PHYS = 0.1
GAMMA = 3e-2
N_MODES_R = 16
N_MODES_T = 16
N_R = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(hidden_channels, n_layers):
    class SPMOperator(nn.Module):
        def __init__(self):
            super().__init__()
            self.fno = FNO(
                n_modes=(N_MODES_R, N_MODES_T),
                in_channels=3,
                out_channels=3,
                hidden_channels=hidden_channels,
                n_layers=n_layers,
            )
        def forward(self, x):
            return self.fno(x)
    return SPMOperator().to(device)


def unnormalise(x, mean, std):
    return x * std + mean


def objective(trial):
    # ── suggest hyperparameters ───────────────────────────────────────────────
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
    hidden_channels = trial.suggest_categorical("hidden_channels", [32, 64, 128])
    n_layers = trial.suggest_int("n_layers", 2, 6)

    # ── data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, _, norm_stats, r = get_dataloaders(FILE_NAME, N_BATCH)
    norm_stats = {k: v.to(device) if torch.is_tensor(v) else v
                  for k, v in norm_stats.items()}
    r = r.to(device)

    # ── model + optimiser ────────────────────────────────────────────────────
    model = build_model(hidden_channels, n_layers)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    mse = nn.MSELoss()

    # ── physics constants ────────────────────────────────────────────────────
    consts = get_spm_const()
    R_n, R_p = consts["R_n"], consts["R_p"]
    D_n, D_p = consts["D_n"], consts["D_p"]
    c0_n, c0_p = consts["c0_n"], consts["c0_p"]
    c_scale_n = consts["c_max_n"]
    c_scale_p = consts["c_max_p"]
    r_phys_n = (r * R_n).to(device)
    r_phys_p = (r * R_p).to(device)
    dr_n = physical_grid(R_n, N_R, 1.0, 2)[0]
    dr_p = physical_grid(R_p, N_R, 1.0, 2)[0]
    reg_weights = {"pde": 1.0, "bc_surf": 1.0, "bc_centre": 1.0, "ic": 1.0}

    best_val_loss = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        for batch in train_loader:
            I, t, c_n, c_p, phi_p, j_n, j_p, t_end = [x.to(device) for x in batch]

            x = model_input(I, r, t)
            y_pred = model(x)
            c_n_pred, c_p_pred, phi_p_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

            loss_data = mse(c_n_pred, c_n) + mse(c_p_pred, c_p) + mse(phi_p_pred, phi_p)

            c_n_phys = unnormalise(c_n_pred, norm_stats["c_n_mean"], norm_stats["c_n_std"])
            c_p_phys = unnormalise(c_p_pred, norm_stats["c_p_mean"], norm_stats["c_p_std"])
            n_t = I.shape[-1]
            dt = t_end / (n_t - 1)

            loss_phys, _ = physics_loss(
                c_p_phys, c_n_phys, j_p[:, -1, :], j_n[:, -1, :],
                r_phys_p, r_phys_n, dr_p, dr_n, D_p, D_n,
                c0_p, c0_n, dt, c_scale_p, c_scale_n, R_p, R_n,
                reg_weights, gamma=GAMMA, use_temporal_causality=True
            )

            loss = loss_data + LAMBDA_PHYS * loss_phys
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # ── validate ──────────────────────────────────────────────────────
        model.eval()
        val_data_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                I, t, c_n, c_p, phi_p, j_n, j_p, t_end = [x.to(device) for x in batch]
                x = model_input(I, r, t)
                y_pred = model(x)
                c_n_pred, c_p_pred, phi_p_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]
                val_data_loss += (mse(c_n_pred, c_n) + mse(c_p_pred, c_p)
                                  + mse(phi_p_pred, phi_p)).item()
                n_batches += 1
        val_data_loss /= n_batches

        best_val_loss = min(best_val_loss, val_data_loss)

        # report to optuna and check if this trial should be pruned
        trial.report(val_data_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


if __name__ == "__main__":
    # MedianPruner: after warmup_steps epochs, prune if this trial's val loss
    # is worse than the median of all completed trials at the same step.
    # n_startup_trials: don't prune until at least this many trials have
    # completed (gives the pruner enough data to compute a meaningful median).
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=WARMUP_EPOCHS,
        interval_steps=5,   # check for pruning every 5 epochs, not every 1
    )

    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="spm_pino_sweep",
        storage="sqlite:///spm_sweep.db",   # persists results to disk --
                                             # if the sweep crashes, you can
                                             # resume without losing completed
                                             # trials by re-running this script
        load_if_exists=True,                 # resume a partially-done sweep
    )

    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n=== Sweep complete ===")
    print(f"Best val loss: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # show top 10 trials for reference
    print("\nTop 10 trials:")
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.value)
    for i, t in enumerate(completed[:10]):
        print(f"  #{i+1} val={t.value:.4f} | {t.params}")