"""
https://www.geeksforgeeks.org/deep-learning/transformer-using-pytorch/
"""

import torch 
from torch import nn
import math

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
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # self.d_model = d_model
        # self.num_heads = num_heads
        # self.d_k = d_model // num_heads
        # self.W_q = nn.Linear(d_model, d_model)
        # self.W_k = nn.Linear(d_model, d_model)
        # self.W_v = nn.Linear(d_model, d_model)
        # self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # if mask is not None:
        #     attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        # attn_probs = torch.softmax(attn_scores, dim=-1)
        # output = torch.matmul(attn_probs, V)
        # return output

    def split_heads(self, x):
        # batch_size, seq_length, d_model = x.size()
        # return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        # batch_size, _, seq_length, d_k = x.size()
        # return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # Q = self.split_heads(self.W_q(Q))
        # K = self.split_heads(self.W_k(K))
        # V = self.split_heads(self.W_v(V))
        # attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        # output = self.W_o(self.combine_heads(attn_output))
        # return output


if __name__ == "__main__":
    main()

