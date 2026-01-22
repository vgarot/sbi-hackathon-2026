import torch.nn as nn
import torch

class SingleCoupling(nn.Module):
    def __init__(self,
                 dim:int,
                 hidden_dim:int,
                 ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        x1 = x * self.mask
        x2 = x * (1 - self.mask)

        s_t = self.net(x1)
        y2 = x2 + s_t

        y = x1 + y2

        return y