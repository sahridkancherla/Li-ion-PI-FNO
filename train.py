"""Training the PINO"""
import numpy as np
import torch
import torch.nn as nn

from data_loader import get_dataloaders
from model import init_model, device
from spm_constants import get_spm_const
from physics_loss import physics_loss, pde_residual_per_timestep, temporal_causal_weights
from utils import model_input, physical_grid

# Config
file_name = "spm_data_v3.npz"
n_batch = 32
n_epochs = 100
lr = 1e-3
lambda_phys = 0.1
gamma = 8e-3          # temporal-causality strength on the PDE residual (Wang et al. 2022)
                     # gamma=0 recovers the plain unweighted PDE loss
n_r = 32

# Setup
train_loader, val_loader, test_loader, norm_stats, r = get_dataloaders(file_name, n_batch)
norm_stats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in norm_stats.items()}
r = r.to(device)

pino = init_model()
optimiser = torch.optim.AdamW(pino.parameters(), lr=lr, weight_decay=1e-2)
mse = nn.MSELoss()

consts = get_spm_const()
R_n, R_p = consts["R_n"], consts["R_p"]
D_n, D_p = consts["D_n"], consts["D_p"]
c0_n, c0_p = consts["c0_n"], consts["c0_p"]
c_scale_n, c_scale_p = consts["c_max_n"], consts["c_max_p"]

# physical radial coords and physical spacing for derivatives
r_phys_n = (r * R_n).to(device)
r_phys_p = (r * R_p).to(device)
# params for dt (1.0, 2) below are just placeholders, since dt computed per batch later in loop
dr_n, dr_p = physical_grid(R_n, n_r, 1.0, 2)[0], physical_grid(R_p, n_r, 1.0, 2)[0]

def unnormalise(x, mean, std):
    return x * std + mean

def run_epoch(loader, train=True):
    pino.train() if train else pino.eval()
    total_loss, total_data_loss, total_phys_loss = 0.0, 0.0, 0.0
    n_batches = 0
    reg_weights = {"pde": 1.0, "bc_surf": 1.0, "bc_centre": 1.0, "ic": 1.0}
    last_weight_min, last_weight_max = None, None  # diagnostic: range of temporal-causality
                                                     # weights on the most recent batch

    grad_on = torch.enable_grad() if train else torch.no_grad()
    with grad_on:
        for batch in loader:
            I, t, c_n, c_p, phi_p, j_n, j_p, t_end = [x.to(device) for x in batch]
            batch_size = I.shape[0]

            # forward pass:
            x = model_input(I, r, t) # [batch, 3, n_r, n_t]
            y_pred = pino(x) # [batch, 3, n_r, n_t]
            c_n_pred, c_p_pred, phi_p_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

            # data loss:
            loss_data = (mse(c_n_pred, c_n) +
                         mse(c_p_pred, c_p) +
                        mse(phi_p_pred, phi_p))

            # physics loss (with temporal-causality weighting on the PDE term):
            c_n_pred_phys = unnormalise(c_n_pred, norm_stats["c_n_mean"], norm_stats["c_n_std"])
            c_p_pred_phys = unnormalise(c_p_pred, norm_stats["c_p_mean"], norm_stats["c_p_std"])

            n_t = I.shape[-1]
            dt = t_end / (n_t - 1)

            loss_physics, loss_components = physics_loss(c_p_pred_phys, c_n_pred_phys, j_p[:, -1, :], j_n[:, -1, :],
                                                        r_phys_p, r_phys_n, dr_p, dr_n, D_p, D_n,
                                                        c0_p, c0_n, dt, c_scale_p, c_scale_n, R_p, R_n,
                                                        reg_weights, gamma=gamma, use_temporal_causality=True)

            # diagnostic only -- recompute the weights for the negative-electrode
            # PDE residual on this batch so we can report their range. This is a
            # cheap, no-grad-needed extra computation purely for visibility into
            # whether `gamma` is actually doing anything (weights near 1.0
            # everywhere means gamma is too small to matter on this problem's
            # residual scale).
            with torch.no_grad():
                diag_loss_per_t = pde_residual_per_timestep(c_n_pred_phys, r_phys_n, D_n, dr_n, dt, c_scale_n, R_n)
                diag_weights = temporal_causal_weights(diag_loss_per_t, gamma=gamma)
                last_weight_min = diag_weights.min().item()
                last_weight_max = diag_weights.max().item()

            loss = loss_data + lambda_phys * loss_physics

            # train:
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item()
            total_data_loss += loss_data.item()
            total_phys_loss += loss_physics.item()
            n_batches += 1

    return (total_loss / n_batches, total_data_loss / n_batches, total_phys_loss / n_batches,
            last_weight_min, last_weight_max)


if __name__ == "__main__":
    for epoch in range (1, n_epochs + 1):
        train_loss, train_data_loss, train_physics_loss, w_min, w_max = run_epoch(train_loader, train=True)
        test_loss, test_data_loss, test_physics_loss, _, _ = run_epoch(test_loader, train=False)

        print(f"Epoch {epoch:3d} | "
              f"train: total={train_loss:.4f} data={train_data_loss:.4f} physics={train_physics_loss:.4f} | "
              f"test: total={test_loss:.4f} data={test_data_loss:.4f} physics={test_physics_loss:.4f} | "
              f"causal_weights=[{w_min:.4f}, {w_max:.4f}]")

        delta_loss = test_loss - train_loss

    torch.save({
                "model_state_dict": pino.state_dict(),
                "norm_stats": norm_stats,
                "r": r.cpu(),
            }, "spm_pino_checkpoint_v4_temporal_causality.pt")
    print("Saved checkpoint to spm_pino_checkpoint_v4_temporal_causality.pt")