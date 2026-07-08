"""Shape handling helpers shared between train.py and physics_loss.py"""
import torch

def expand_r(r, batch_size, n_t):
    """Broadcast radial coordinate to match batch of (r, t) fields
    Args:
        r: [n_r] tensor, normalised radial grid. N.B. already on CUDA
        batch_size: int
        n_t: int, no. of time points
    
    Returns:
        [batch_size, n_r, n_t] tensor: same r profile repeated for every sample and timestep
    
    """
    n_r = r.shape[0]
    return r.view(1, n_r, 1).expand(batch_size, n_r, n_t)

def model_input(I, r, t):
    """Stack I(t), r, t into PINO input tensor.

    Args:
        I: [batch, n_r, n_t]
        r: [n_r]
        t: [batch, n_r, n_t]
    
    Returns:
        [batch, 3, n_r, n_t]
    """
    batch_size, n_r, n_t = I.shape
    r_expanded = expand_r(r, batch_size, n_t)
    x = torch.stack([I, r_expanded, t], dim=1)

    return x

def physical_grid(R, n_r, t_end, n_t):
    """Compute physical spacing grid for finite-differene derivative.
    Args:
        R: float or [batch] tensor - particle radius in m
        n_r: int - no. of radial points
        t_end: float or [batch] tensor - simulation duration in s
        n_t: int - no. of temporal points
    
    Returns:
        dr, dt: same type/shape as R, t_end
    """
    dr = R / (n_r - 1)
    dt = t_end / (n_t - 1) 
    return dr, dt

def reshape_dx(dx, f, dim):
    """Broadcast dx against f along differentiation dimension
    dx can be python float / [0-d] tensor (same spacing for all samples) OR [batch] tensor
    i.e. dt varying with duration of each sample. 

    Returns dx reshaped so that it broadcasts correctly when dividing a tensor of f's shape.
    """
    if isinstance(dx, torch.Tensor) and dx.ndim > 0:
        assert dx.shape[0] == f.shape[0], (
            f"dx batch dim {dx.shape[0]} does not match f batch dim {f.shape[0]}"
            )
        shape = [1] * f.ndim
        shape[0] = f.shape[0]
        return dx.view(*shape)
    return dx

class EarlyStopping:
    """Stops training when val loss hasn't improved by more than min_delta
    for `patience` consecutive epochs. Also saves the best checkpoint.

    Args:
        patience:   epochs to wait after last sufficient improvement
        min_delta:  minimum improvement to count as 'real' (filters noise)
        checkpoint_path: where to save the best model
    """
    def __init__(self, patience=30, min_delta=1e-3, checkpoint_path="spm_pino_checkpoint_best.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path

        self.best_loss = float("inf")
        self.epochs_no_improve = 0
        self.best_epoch = 0
        self.should_stop = False
    
    def step(self, val_loss, epoch, model, norm_stats, r):
        """Call once per epoch with the current val loss.
        Returns True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "norm_stats": norm_stats,
                "r": r.cpu(),
                "epoch": epoch,
                "val_loss": val_loss,
            }, self.checkpoint_path)
            return False
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True
                return True
            return False
    
    def summary(self):
        print(f"Early stopping triggered. Best val loss {self.best_loss:.4f} "
              f"at epoch {self.best_epoch}. "
              f"Checkpoint saved to {self.checkpoint_path}")
