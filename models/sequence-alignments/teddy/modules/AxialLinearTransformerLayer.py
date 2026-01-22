import torch.nn as nn
from teddy.modules.AxialColumnLinearAttention import MultiheadFlexAttention

class AxialLinearTransformerLayer(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 nb_heads: int,
                 dropout: float,
                 fnn_dim: int,
                 normalization:bool = True,
                 ):
        super().__init__()

        self.nb_heads = nb_heads
        self.row_attn = MultiheadFlexAttention(embed_dim, embed_dim, nb_heads)
        self.col_attn = MultiheadFlexAttention(embed_dim, embed_dim, nb_heads)

        self.norm_row = nn.LayerNorm(embed_dim)
        self.norm_col = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim,fnn_dim)
        self.fnndropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fnn_dim, embed_dim)
        self.activ_ffn = nn.GELU()

        self.normalization = normalization
        if normalization:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self,input, block_mask = None):
        B, N, C , D = input.size()
        block_mask_row, block_mask_column = block_mask
        
        y = input.transpose(-2,-3).reshape(B*C, N, D)

        after_row = self.row_attn(y, block_mask_row)
        after_row = after_row.reshape(B, C, N, D).transpose(-2,-3) + input
        after_row = self.norm_row(after_row)

        z = after_row.reshape(B*N, C, D)

        after_col = self.col_attn(z, block_mask_column)
        after_col = after_col.reshape(B,N, C, D) + after_row
        after_col = self.norm_col(after_col)

        ffn_x = self.activ_ffn(self.fc1(after_col))
        ffn_x = self.fnndropout(ffn_x)
        ffn_x = self.fc2(ffn_x) + after_col
        if self.normalization:
            ffn_x = self.norm(ffn_x)
        return ffn_x