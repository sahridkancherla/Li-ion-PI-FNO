"""Diagnose whether the trained FNO violates causality.

Test: take real test samples, randomise ssecond half of I(t) but keep the first
half unchanged. If predictions for first half change as a
result, the model is using future current information to predict past states.
If first-half predictions barely change, causality likely isn't the dominant
cause of poor generalization.
"""
import torch

from data_loader import get_dataloaders
from model import init_model, device
from utils import model_input

FILE_NAME = "spm_data_v2.npz"
N_BATCH = 32
CHECKPOINT_PATH = "spm_pino_checkpoint_v3_temporal_causality.pt"


def run_causality_diagnostic():
    train_loader, val_loader, test_loader, norm_stats, r = get_dataloaders(FILE_NAME, N_BATCH)
    norm_stats = {k: v.to(device) if torch.is_tensor(v) else v for k, v in norm_stats.items()}
    r = r.to(device)

    model = init_model()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict)
    model.eval()

    batch = next(iter(test_loader))
    I, t, c_n, c_p, phi_p, j_n, j_p, t_end = [x.to(device) for x in batch]
    n_t = I.shape[-1]
    half = n_t // 2

    with torch.no_grad():
        # ── baseline prediction, unmodified input ──────────────────────────
        x_orig = model_input(I, r, t)
        y_orig = model(x_orig)

        # ── corrupted prediction: randomise I in the SECOND half only ──────
        I_corrupt = I.clone()
        I_corrupt[:, :, half:] = torch.randn_like(I_corrupt[:, :, half:])
        x_corrupt = model_input(I_corrupt, r, t)
        y_corrupt = model(x_corrupt)

    # compare predictions in the FIRST half (should be IDENTICAL if causal)
    diff_first_half = (y_orig[:, :, :, :half] - y_corrupt[:, :, :, :half])
    diff_second_half = (y_orig[:, :, :, half:] - y_corrupt[:, :, :, half:])

    rmse_first = (diff_first_half ** 2).mean().sqrt().item()
    rmse_second = (diff_second_half ** 2).mean().sqrt().item()

    # also get a reference scale: typical magnitude of predictions themselves
    pred_scale = y_orig.std().item()

    print("=== Causality diagnostic ===")
    print(f"Prediction std (reference scale): {pred_scale:.6f}")
    print(f"RMSE change in FIRST half (t < t_corrupt):  {rmse_first:.6f}  "
          f"({100*rmse_first/pred_scale:.2f}% of pred scale)")
    print(f"RMSE change in SECOND half (t >= t_corrupt): {rmse_second:.6f}  "
          f"({100*rmse_second/pred_scale:.2f}% of pred scale)")
    print()
    if rmse_first / pred_scale > 0.01:
        print("=> Causality violation DETECTED: fisrst-half predictions change "
              "meaningfully when only future current is altered.")
    else:
        print("=> No significant causality violation detected: first-half "
              "predictions are essentially unchanged.")


if __name__ == "__main__":
    run_causality_diagnostic()