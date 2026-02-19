import torch
import torch.nn as nn
import torch.nn.functional as F
import math




# def create_mask(size):
#     r"""
#     Step A: The Mask (The "No Peeking" Rule)This is what makes the transformer "autoregressive." 
#     We create a square matrix where the upper triangle is filled with $-\infty$.    
#     """
#     # torch.triu returns the upper triangular part of a matrix
#     mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
#     # Mask looks like:
#     # [[F, T, T],
#     #  [F, F, T],
#     #  [F, F, F]]  -> True means "hide this"
#     return mask



# # Convert word IDs to vectors
# x = self.embedding(input_tokens) 

# # Add sine/cosine waves so the model knows the order of words
# x = x + self.positional_encoding(x)


# # 1. Multi-Head Attention: "What words relate to each other?"
# # We pass the mask here to ensure we don't look ahead.
# attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)

# # 2. Residual Connection & Norm: "Keep the original info and clean it up."
# x = self.norm1(x + attn_output)

# # 3. Feed Forward: "Think about the relationships found in step 1."
# ff_output = self.feed_forward(x)
# x = self.norm2(x + ff_output)


# # Project the results to the size of the vocabulary (e.g., 50,000 words)
# logits = self.fc_out(x)

# # During training, we use Cross-Entropy Loss:
# # Loss = CrossEntropy(logits, target_tokens)



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super.__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)




















# --- Step 1: Positional Encoding ---
# Since Transformers process all tokens at once, they need a way to "know" word order.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# # --- Step 2: The Transformer Block ---
# # This contains the Masked Multi-Head Attention and the Feed Forward Network.
# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
#         super().__init__()
#         self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(d_model, dim_feedforward),
#             nn.ReLU(),
#             nn.Linear(dim_feedforward, d_model)
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x, mask):
#         # Masked Self-Attention
#         attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         # Feed Forward
#         ff_output = self.feed_forward(x)
#         x = self.norm2(x + self.dropout(ff_output))
#         return x

# # --- Step 3: The Full Autoregressive Model ---
# class AutoregressiveTransformer(nn.Module):
#     def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len):
#         super().__init__()
#         self.embedding = nn.Embedding(vocab_size, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, max_len)
#         self.layers = nn.ModuleList([
#             TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)
#         ])
#         self.fc_out = nn.Linear(d_model, vocab_size)

#     def generate_mask(self, sz):
#         # Creates a lower-triangular mask to prevent attending to future tokens
#         mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
#         return mask

#     def forward(self, x):
#         sz = x.size(1)
#         mask = self.generate_mask(sz).to(x.device)
        
#         x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
#         x = self.pos_encoder(x)
        
#         for layer in self.layers:
#             x = layer(x, mask)
            
#         return self.fc_out(x)