import torch.nn as nn
import torch
import math

class TimeEncoding(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 ):
        super().__init__()
        self.register_buffer("div_term",torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)))
    
    def forward(self,x,dates):
        theta = torch.einsum('br,d->brd',dates,self.div_term)
        x[:,1:,0,0::2] += torch.sin(theta)
        x[:,1:,0,1::2] += torch.cos(theta)
        return x