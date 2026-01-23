import torch.nn as nn
import torch
from torch.nn.attention.flex_attention import flex_attention


# Wrapper function for flex_attention that can be compiled
def _flex_attention_wrapper(queries, keys, values, block_mask):
    """Wrapper for flex_attention that is compatible with torch.compile()"""
    return flex_attention(queries, keys, values, block_mask=block_mask)


# Compile the wrapper function
_compiled_flex_attention = torch.compile(_flex_attention_wrapper, mode="reduce-overhead", fullgraph=False)


class MultiheadFlexAttention(nn.Module):
    def __init__(self, d_in, d_out, n_heads, bias=False):
        """
        description: a torch module that implements multiheaded self-attention via flex_attention.
        args:
            d_in: int, the dimension of the input tensor.
            d_out: int, the dimension of the output tensor.
            n_heads: int, the number of heads to use for the multiheaded self-attention.
            bias: bool, whether to use query, key, and value biases
        """
        super().__init__()
        assert d_out % n_heads == 0, "d_out must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.d_out = d_out

        self.in_proj = nn.Linear(d_in, 3 * d_out, bias=bias)
        self.out_proj = nn.Linear(d_out, d_out)

    def forward(self, x, block_mask):
        """
        description: forward pass of the multiheaded self-attention module.
        args:
            x: torch.Tensor, the input tensor of shape (batch_size, max_seq_len, d_in)
            block_mask: torch.Tensor, the block mask to use for flex_attention
        """
        batch_size, max_seq_len, d_in = x.shape

        # Create stacked qkv via input projection
        qkv = self.in_proj(x) # (batch_size, max_seq_len , 3 * d_in)

        # Split qkv and divide d_in into heads
        qkv = qkv.view(batch_size, max_seq_len, 3, self.n_heads, self.d_head) # (batch_size, max_seq_len, 3, n_heads, d_head)

        # Permute shape of qkv for flex_attention
        qkv = qkv.permute(2, 0, 3, 1, 4) # (3, batch_size, n_heads, max_seq_len, d_head)

        # Get queries, keys, values
        queries, keys, values = qkv # 3 x (batch_size, n_heads, max_seq_len, d_head)
        # Calculate attention via compiled flex_attention
        attn = _compiled_flex_attention(queries, keys, values, block_mask) # (batch_size, n_heads, max_seq_len, d_head)

        # Merge heads into d_out
        attn = attn.transpose(1, 2).contiguous().view(batch_size, max_seq_len, self.d_out)

        # Pass attention output through output projection
        attn = self.out_proj(attn)

        return attn
