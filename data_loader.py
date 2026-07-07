"""Loads data from .npz files and returns PyTorch DataLoaders for training, validation, and testing."""
import numpy as np
import torch

# Load data
def load_data(file_path):
    data = np.load(file_path)
    r = torch.tensor(data["r_norm"], dtype=torch.float32)  # [n_r]

    def to_tensor(key):
        return torch.tensor(data[key], dtype=torch.float32)

    # tuple order: I, t, c_sn, c_sp, phi_sp, j_n, j_p, t_end
    train_data = (to_tensor("I_train"), to_tensor("t_train"), to_tensor("c_sn_train"),
                  to_tensor("c_sp_train"), to_tensor("phi_sp_train"),
                  to_tensor("j_n_train"), to_tensor("j_p_train"), to_tensor("t_end_train"))
    val_data   = (to_tensor("I_val"),   to_tensor("t_val"),   to_tensor("c_sn_val"),
                  to_tensor("c_sp_val"),   to_tensor("phi_sp_val"),
                  to_tensor("j_n_val"),   to_tensor("j_p_val"),   to_tensor("t_end_val"))
    test_data  = (to_tensor("I_test"),  to_tensor("t_test"),  to_tensor("c_sn_test"),
                  to_tensor("c_sp_test"),  to_tensor("phi_sp_test"),
                  to_tensor("j_n_test"),  to_tensor("j_p_test"),  to_tensor("t_end_test"))

    return r, train_data, val_data, test_data


def get_dataloaders(file_path, batch_size=32):
    r, train_data, val_data, test_data = load_data(file_path)
    # Normalisation for training data only
    I_mean, I_std = train_data[0].mean(), train_data[0].std()
    c_sn_mean, c_sn_std = train_data[2].mean(), train_data[2].std()
    c_sp_mean, c_sp_std = train_data[3].mean(), train_data[3].std()
    phi_sp_mean, phi_sp_std = train_data[4].mean(), train_data[4].std()

    norm_stats = {
    "I_mean": I_mean, "I_std": I_std,
    "c_n_mean": c_sn_mean, "c_n_std": c_sn_std,
    "c_p_mean": c_sp_mean, "c_p_std": c_sp_std,
    "phi_p_mean": phi_sp_mean, "phi_p_std": phi_sp_std,
    }

    def normalise(data, I_mean, I_std, c_n_mean, c_n_std,
                  c_p_mean, c_p_std, phi_p_mean, phi_p_std):
        return (
            (data[0] - I_mean) / I_std, # I
             data[1], # t, already normalised
            (data[2] - c_n_mean) / c_n_std, # c_sn
            (data[3] - c_p_mean) / c_p_std, # c_sp
            (data[4] - phi_p_mean) / phi_p_std, # phi_sp
            data[5], # j_n, A/m^2 - NOT normalised, needed for BC residual
            data[6], # j_p, A/m^2 - NOT normalised, needed  for BC residual
            data[7], # t_end, raw seconds - NOT normalised, needed for physics loss dt
        )

    train_data = normalise(train_data, **norm_stats)
    val_data = normalise(val_data, **norm_stats)
    test_data = normalise(test_data, **norm_stats)

    def make_loader(data, shuffle=True):
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*data),
            batch_size=batch_size, shuffle=shuffle,
            pin_memory=True, num_workers=0
        )
    
    train_loader = make_loader(train_data, shuffle=True)
    val_loader = make_loader(val_data, shuffle=False)
    test_loader = make_loader(test_data, shuffle=False)

    return train_loader, val_loader, test_loader, norm_stats, r