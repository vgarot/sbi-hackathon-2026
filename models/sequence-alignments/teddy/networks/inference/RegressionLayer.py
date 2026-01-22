import torch.nn as nn

class RegressionLayer(nn.Module):
  
    def __init__(self,
                avg_dim: int,
                output_dim: int,
                hidden_dim:int,
                ):
        super().__init__()
        self.fc1 = nn.Linear(avg_dim,hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,output_dim)
        self.activ = nn.ReLU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.softplus = nn.Softplus()

    def forward(self, input):
        output = self.norm(self.activ(self.fc1(input)))
        output = self.fc2(output)

        output = self.softplus(output)
        output = output.cumsum(dim=-1)
        output[:,3:] = output[:,3:] - output[:,2].unsqueeze(1)
        
        return output