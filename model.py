import torch
from neuralop.models import FNO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SPMOperator(torch.nn.Module):
    def __init__(self, n_modes_r=16, n_modes_t=16, hidden_channels=64, n_layers=4):
        super().__init__()
        self.fno = FNO(
            n_modes=(n_modes_r, n_modes_t),  
            in_channels=3,                    # I(t), r, t
            out_channels=3,                   # c_sn, c_sp, phi_sp
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            use_channel_mlp=True,
            channel_mlp_dropout=0.1
        )

    def forward(self, x):
        # x: [batch, 3, N_R, N_T]
        return self.fno(x)  # [batch, 3, N_R, N_T]

def init_model():
    return SPMOperator().to(device)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
