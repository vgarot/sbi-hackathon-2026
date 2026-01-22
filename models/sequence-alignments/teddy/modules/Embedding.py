import torch.nn as nn
from teddy.modules.TimeEncoding import TimeEncoding
from torch.nn.attention.flex_attention import create_block_mask


class Embedding(nn.Module):
    def __init__(self,
                 alphabet,
                 embed_dim:int,
                 consider_dates:bool = True,
                 ):
        super().__init__()
        self.alphabet_size = len(alphabet)
        self.consider_dates = consider_dates
        self.embed_dim = embed_dim

        self.idx_padding = alphabet.pad_idx
        self.s_idx = alphabet.s_idx
        self.seq_idx = alphabet.seq_idx
        self.site_idx = alphabet.site_idx
        self.nonconstant_site_idx = alphabet.nonconstant_site_idx


        self.time_encoding = TimeEncoding(embed_dim)

        self.linear1 = nn.Linear(self.alphabet_size, embed_dim)
        self.relu = nn.ReLU()
        self.layernorm = nn.LayerNorm(embed_dim)


    def forward(self,input):
        def create_padding_mask(pads):
            def padding(b, h, q_idx, kv_idx):
                return ~pads[b, q_idx] & ~pads[b, kv_idx]
            return padding

        x = input[0]
        shapes = input[1]

        dates, data, s = x[:,1:,0], x[:,:,1:], x[:,0,0] # BATCH, ROW, COL
        
        mask = (data == self.idx_padding)

        data = nn.functional.one_hot(data.long(),num_classes=self.alphabet_size).float()
        s = s.unsqueeze(-1).unsqueeze(-1).expand(-1,data.shape[1],data.shape[2])
        seq = shapes[:,0].unsqueeze(-1).unsqueeze(-1).expand(-1,data.shape[1],data.shape[2])/100
        sites = shapes[:,1].unsqueeze(-1).unsqueeze(-1).expand(-1,data.shape[1],data.shape[2])/1000
        nonconstant_sites = shapes[:,2].unsqueeze(-1).unsqueeze(-1).expand(-1,data.shape[1],data.shape[2])/1000

        data[:,:,:,self.s_idx] = s
        data[:,:,:,self.seq_idx] = seq
        data[:,:,:,self.site_idx] = sites
        data[:,:,:,self.nonconstant_site_idx] = nonconstant_sites

        embedding = self.linear1(data)
        embedding = self.relu(embedding)
        embedding = self.layernorm(embedding)

        if self.consider_dates:
            embedding = self.time_encoding(embedding,dates)
        
        # create block masks for flex
        B, N, C , D = embedding.size()
        mask_row = mask.transpose(-1,-2).reshape(B*C, N).contiguous()
        mask_column = mask.reshape(B*N, C).contiguous()
        padd_mask_row = create_padding_mask(mask_row)
        padd_mask_column = create_padding_mask(mask_column)
        
        block_mask_row = create_block_mask(padd_mask_row,
                                           B = B*C,
                                           H = None,
                                           Q_LEN=N,
                                           KV_LEN=N,
                                           _compile=True,
                                           device=embedding.device
                                           )
        block_mask_column = create_block_mask(padd_mask_column,
                                              B = B*N,
                                              H = None,
                                              Q_LEN=C,
                                              KV_LEN=C,
                                              _compile=True,
                                              device=embedding.device
                                              )
        
        return embedding, block_mask_row, block_mask_column