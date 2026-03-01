import torch
import torch.nn as nn
import torch.nn.functional as F
import math





class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super.__init__()
        constant = 10000
        pe = torch.zeros(max_len, d_model)
        positions = torch.range(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,max_len,2) * (-math.log(constant) / d_model))
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)





#! implementing ROPE

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # dim is the head_dim
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    
    # We convert to complex numbers for a more efficient 'rotation' 
    # via complex multiplication: e^(i*theta)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x shape: [batch, seq_len, n_heads, head_dim]
    # 1. Reshape x to treat the last dimension as pairs (complex numbers)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    
    # 2. Reshape freqs_cis to match the broadcast shape
    # freqs_cis shape: [seq_len, head_dim // 2]
    freqs_cis = freqs_cis.view(1, x.shape[1], 1, -1)
    
    # 3. Rotate by multiplying complex numbers
    x_rotated = x_complex * freqs_cis
    
    # 4. Flatten back to real numbers
    return torch.view_as_real(x_rotated).flatten(3)



# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super.__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # (max_len, 1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0)) # this is a constant





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

# --- Step 2: The Transformer Block ---
# This contains the Masked Multi-Head Attention and the Feed Forward Network.
class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Masked Self-Attention
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed Forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

# --- Step 3: The Full Autoregressive Model ---
class AutoregressiveTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_mask(self, sz):
        # Creates a lower-triangular mask to prevent attending to future tokens
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def forward(self, x):
        sz = x.size(1)
        mask = self.generate_mask(sz).to(x.device)
        
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.fc_out(x)