"""
https://www.geeksforgeeks.org/deep-learning/transformer-using-pytorch/


Input
 ↓
[ Multi-Head Self-Attention ]
 ↓
+ Residual + LayerNorm
 ↓
[ Feed-Forward Network ]
 ↓
+ Residual + LayerNorm
 ↓
Output

"""

import torch 
from torch import nn
import math
from typing import Optional

def softmax(x, dim=-1):
    """
    x: Tensor of shape (..., num_classes)
    dim: dimension along which softmax is computed
    """
    # 1. Numerical stability: subtract max
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - x_max

    # 2. Exponentiate
    exp_x = torch.exp(x_stable)

    # 3. Normalize
    softmax_x = exp_x / torch.sum(exp_x, dim=dim, keepdim=True)

    return softmax_x



# class SelfAttention(torch.nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  hidden_dim: int) -> None:
#         super().__init__()     
#         self.Ql = nn.Linear(in_dim, hidden_dim)
#         self.Kl = nn.Linear(in_dim, hidden_dim)
#         self.Vl = nn.Linear(in_dim, hidden_dim)
#         self.norm = math.sqrt(hidden_dim) # root(d)

#     def forward(self, x):
#         q = self.Ql(x) # query
#         v = self.Vl(x) # value
#         k = self.Kl(x) # key
#         atten = softmax(q @ k.t() / self.norm)
#         assert torch.all((x >= 0) & (x <= 1)).item()
#         out = atten @ v
#         return out


# class MultiHead(torch.nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  hidden_dim: int,
#                  num_heads: int) -> None:
#         super().__init__()     
#         self.heads = []
#         self.num_heads = num_heads
#         self.out_layer = nn.Linear(hidden_dim, hidden_dim)
#         for i in range(num_heads):
#             self.heads.append(SelfAttention(in_dim, hidden_dim))
    
#     def forward(self, x):
#         heads = [self.out_layer(head(x)) for head in self.heads]
#         heads = torch.cat(heads, dim=1)
#         return heads
        



# def main():
#     num_nodes = 100
#     in_dim = 64
#     hidden_dim = 64

    
#     x = torch.rand(num_nodes, in_dim)
#     print (x[0])
#     # atten = SelfAttention(in_dim=in_dim, hidden_dim=hidden_dim)
#     # out = atten(x)
#     # print (out[0])

#     num_heads = 8
#     mh = MultiHead(in_dim=in_dim, hidden_dim=hidden_dim, num_heads=num_heads)
#     mh_out = mh(x)
#     assert mh_out.shape == (num_nodes, num_heads*hidden_dim)
#     print (mh_out[0])


class MultiHeadAttention(nn.Module):
    """
    This block defines the MultiHeadAttention class. It splits the input into multiple attention heads, computes scaled dot-product attention, and then combines the outputs.
    """
    def __init__(self, 
                 d_model:int, 
                 num_heads:int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads #because d_model is the final concatenated output size from all heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Q: # (B, num_heads, seq, d_k) 
        K: # (B, num_heads, seq, d_k) 
        K.transpose(-2, -1): # (B, num_heads, d_k, seq) 

        (batch1, batch2, M, K) @ (batch1, batch2, K, N) → (batch1, batch2, M, N)
        """
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, num_heads, seq, seq)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)  
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # attn_probs: [B, num_heads, seq, seq]
        # V: [B, num_heads, seq, d_k]
        output = torch.matmul(attn_probs, V) 
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()  #! B \times seq \times d_model from (B \times seq, d_model)
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2) # (B, seq, num_heads, d_k) --> (B, num_heads, seq, d_k)

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size() # (B, num_heads, seq, d_k)
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Q: B \times seq \times d_model, before
        Q = self.split_heads(self.W_q(Q))  # (B, num_heads, seq, d_k) 
        K = self.split_heads(self.W_k(K))  # (B, num_heads, seq, d_k) 
        V = self.split_heads(self.W_v(V))  # (B, num_heads, seq, d_k) 
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        attn_w = self.combine_heads(attn_output)
        output = self.W_o(attn_w)
        return output, attn_w


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        # x: (batch, seq, embed_dim)
        x = self.fc1(x) # x = xW + b
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x) 
        return x




# ---------- Transformer Block ----------
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ff_hidden_dim: Optional[int] = None,
                 dropout: float = 0.0, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.ff = FeedForward(embed_dim, hidden_dim=ff_hidden_dim)
        self.ln2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        x: (batch, seq_len, embed_dim)
        mask: optional boolean mask broadcastable to (batch, heads, seq, seq) or (batch, 1, 1, seq)
        """
        # Multi-head attention + residual + layernorm
        attn_out, attn_weights = self.attn(x, mask=mask)  # (batch, seq, embed_dim)
        x = x + self.dropout(attn_out) # residual connection
        x = self.ln1(x)

        # Feed-forward + residual + layernorm
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)

        return x, attn_weights



def main():
    batch_size = 100
    seq_len = 256
    hidden_dim = 1024
    num_heads = 8

    
    x = torch.rand(batch_size, seq_len, hidden_dim)

    num_heads = 8
    mh = MultiHeadAttention(d_model=hidden_dim, num_heads=num_heads)
    out, attn_w = mh(x,x,x)
    print (out.shape)
    print (attn_w.shape)
    assert out.shape == (batch_size, seq_len, hidden_dim)
    print (out[0])

if __name__ == "__main__":
    main()

