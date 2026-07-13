"""Training the PINO"""
import numpy as np
import torch
import torch.nn as nn

from data_loader import get_dataloaders
from model import init_model, device
from spm_constants import get_spm_const
from physics_loss import physics_loss, pde_residual_per_timestep, temporal_causal_weights
from utils import model_input, physical_grid, EarlyStopping

# Config
file_name = "spm_data_v3.npz"
n_batch = 32
n_epochs = 200
lr = 1.241e-4
n_r = 32

# Setup
train_loader, val_loader, test_loader, norm_stats, r = get_dataloaders(file_name, n_batch)
norm_stats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in norm_stats.items()}
r = r.to(device)

pino = init_model()
optimiser = torch.optim.AdamW(pino.parameters(), lr=lr, weight_decay=9.5e-4)
mse = nn.MSELoss()


def unnormalise(x, mean, std):
    return x * std + mean

def run_epoch(loader, train=True):
    pino.train() if train else pino.eval()
    total_loss = 0.0
    n_batches = 0

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

            loss = loss_data

            # train:
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            total_loss += loss.item()
            n_batches += 1

    return (total_loss / n_batches)


if __name__ == "__main__":
    early_stopping = EarlyStopping(patience=15, min_delta=5e-4, checkpoint_path="spm_fno_checkpoint.pt")

    for epoch in range (1, n_epochs + 1):
        train_loss = run_epoch(train_loader, train=True)
        test_loss = run_epoch(test_loader, train=False)

        print(f"Epoch {epoch:3d} | "
          f"train: total={train_loss:.4f} | "
          f"test: total={test_loss:.4f} | "
          f"patience={early_stopping.epochs_no_improve}/{early_stopping.patience}")

        if early_stopping.step(test_loss, epoch, pino, norm_stats, r):
            early_stopping.summary()
            break

    print(f"\nBest checkpoint: epoch {early_stopping.best_epoch}, "
        f"test_loss={early_stopping.best_loss:.4f}")