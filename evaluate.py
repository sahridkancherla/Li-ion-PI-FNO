import numpy as np
import torch

from data_loader import get_dataloaders
from model import init_model, device
from spm_constants import get_spm_const
from utils import model_input

file_name = "spm_data_v3.npz"
n_batch = 32
checkpt_path = "spm_pino_checkpoint_v4_temporal_causality.pt"


def unnormalise(x, mean, std):
    return x * std + mean


def compute_rmse(loader, model, norm_stats, r):
    """Compute model RMSE in physical units for c_n, c_p, phi_p over a full loader."""
    model.eval()

    sq_err_c_n, sq_err_c_p, sq_err_phi_p = 0.0, 0.0, 0.0
    n_elements = 0

    with torch.no_grad():
        for batch in loader:
            I, t, c_n, c_p, phi_p, j_n, j_p, t_end = [x.to(device) for x in batch]

            x = model_input(I, r, t)
            y_pred = model(x)
            c_n_pred, c_p_pred, phi_p_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

            # unnormalise both pred and target before computing error
            c_n_pred_phys = unnormalise(c_n_pred, norm_stats["c_n_mean"], norm_stats["c_n_std"])
            c_n_true_phys = unnormalise(c_n, norm_stats["c_n_mean"], norm_stats["c_n_std"])

            c_p_pred_phys = unnormalise(c_p_pred, norm_stats["c_p_mean"], norm_stats["c_p_std"])
            c_p_true_phys = unnormalise(c_p, norm_stats["c_p_mean"], norm_stats["c_p_std"])

            phi_p_pred_phys = unnormalise(phi_p_pred, norm_stats["phi_p_mean"], norm_stats["phi_p_std"])
            phi_p_true_phys = unnormalise(phi_p, norm_stats["phi_p_mean"], norm_stats["phi_p_std"])

            sq_err_c_n += ((c_n_pred_phys - c_n_true_phys) ** 2).sum().item()
            sq_err_c_p += ((c_p_pred_phys - c_p_true_phys) ** 2).sum().item()
            sq_err_phi_p += ((phi_p_pred_phys - phi_p_true_phys) ** 2).sum().item()
            n_elements += c_n.numel()

    rmse_c_n = np.sqrt(sq_err_c_n / n_elements)
    rmse_c_p = np.sqrt(sq_err_c_p / n_elements)
    rmse_phi_p = np.sqrt(sq_err_phi_p / n_elements)

    return rmse_c_n, rmse_c_p, rmse_phi_p


if __name__ == "__main__":
    train_loader, val_loader, test_loader, norm_stats, r = get_dataloaders(file_name, n_batch)
    norm_stats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in norm_stats.items()}
    r = r.to(device)

    pino = init_model()
    checkpt = torch.load(checkpt_path, map_location=device, weights_only=False)
    state_dict = checkpt["model_state_dict"]
    state_dict.pop("_metadata", None)
    pino.load_state_dict(state_dict)

    # reference scales for percentage RMSE -- c_max is the natural physical
    # scale for concentration error (e.g. RMSE / c_max_n as a fraction of the
    # full stoichiometric range). For phi_sp, voltage doesn't have an
    # equivalent single physical constant, so we use the typical operating
    # window of the cell (cutoff to nominal max) as the reference instead.
    consts = get_spm_const()
    c_max_n, c_max_p = consts["c_max_n"], consts["c_max_p"]
    PHI_SP_RANGE = 4.2 - 2.5  # approximate usable voltage window for this cell, V

    for name, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        rmse_c_n, rmse_c_p, rmse_phi_p = compute_rmse(loader, pino, norm_stats, r)

        pct_c_n = 100 * rmse_c_n / c_max_n
        pct_c_p = 100 * rmse_c_p / c_max_p
        pct_phi_p = 100 * rmse_phi_p / PHI_SP_RANGE

        print(f"{name:5s} | RMSE c_sn = {rmse_c_n:10.2f} mol/m^3 ({pct_c_n:5.2f}% of c_max_n) | "
              f"RMSE c_sp = {rmse_c_p:10.2f} mol/m^3 ({pct_c_p:5.2f}% of c_max_p) | "
              f"RMSE phi_sp = {rmse_phi_p:.4f} V ({pct_phi_p:5.2f}% of ~1.7V window)")
