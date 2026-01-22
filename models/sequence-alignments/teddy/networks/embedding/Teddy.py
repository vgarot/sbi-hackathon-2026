import torch.nn as nn

from teddy.networks.inference.RegressionLayer import RegressionLayer
from teddy.modules.AxialLinearTransformerLayer import AxialLinearTransformerLayer
from teddy.modules.Embedding import Embedding


class Teddy(nn.Module):
    def __init__(self, 
                 alphabet, 
                 embed_dim:int, 
                 nb_heads:int, 
                 ffn_dim:int, 
                 nb_layers:int,
                 dropout:float=0.0,
                 dates:bool = True,
                 lastnorm:bool = False,
                 ):
        """
        Teddy model

        Args:
        alphabet (Alphabet): Alphabet object
        embed_dim (int): embedding dimension
        nb_heads (int): number of heads
        ffn_dim (int): feed forward dimension in AxialLinearTransformerLayer
        output_dim (int): output dimension
        hidden_dim (int): hidden dimension in regression layer
        nb_layers (int): number of layers
        dropout (float): dropout rate
        dates (bool): whether to include dates
        lastnorm (bool): whether to include normalization in the last layer
        """

        super().__init__()

        self.embedding = Embedding(
            alphabet,
            embed_dim,
            consider_dates=dates,
            )

        self.layers = nn.ModuleList(
            [
                AxialLinearTransformerLayer(
                    embed_dim,
                    nb_heads,
                    dropout,
                    ffn_dim
                    )
                for _ in range(nb_layers -1)
            ] + 
            [AxialLinearTransformerLayer(
                embed_dim,
                nb_heads,
                dropout,
                ffn_dim,
                normalization=lastnorm)
            ]
            )
        


    
    def forward(self,input):


        embedding, block_mask_row, block_mask_column = self.embedding(input)
        
        for layer in self.layers:
            embedding = layer(embedding, block_mask=(block_mask_row, block_mask_column))
        
        embedding = embedding[:,0,0,:].contiguous()

        return embedding 