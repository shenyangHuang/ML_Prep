"""
prepare ML coding interview about einsum
"""

import torch
import torch.nn.functional as F

# Setup dimensions
batch_size = 8
num_heads = 4
seq_len = 16
head_dim = 64

# 1. Create random Query, Key, and Value tensors
# Shape: [Batch, Heads, Sequence, Head_Dimension]
q = torch.randn(batch_size, num_heads, seq_len, head_dim)
k = torch.randn(batch_size, num_heads, seq_len, head_dim)
v = torch.randn(batch_size, num_heads, seq_len, head_dim)

# 2. Calculate Attention Scores (Q @ K^T)
# We want to multiply over the 'head_dim' (d) but keep 'seq_len' (i and j)
# Logic: [b, h, i, d] * [b, h, j, d] -> [b, h, i, j]
scores = torch.einsum("bhid, bhjd -> bhij", q, k)

# 3. Apply Softmax (standard normalization)
scaling = head_dim ** 0.5
attn_weights = F.softmax(scores / scaling, dim=-1)

# 4. Calculate Context Vector (Scores @ V)
# Logic: [b, h, i, j] * [b, h, j, d] -> [b, h, i, d]
# Here 'j' is the sequence dimension we are summing over.
out = torch.einsum("bhij, bhjd -> bhid", attn_weights, v)

print(f"Output shape: {out.shape}") # Expected: [8, 4, 16, 64]