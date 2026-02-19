"""
an autoregressive transformer
GPT 
bug fix 


The Challenge: "The Future-Seer" GPT
You are given a implementation of a single GPT Transformer Block. 
1. The code runs without crashing, 
2. but the model is "cheating"—the training loss drops to nearly zero instantly, 
3. but the model performs terribly on actual text generation.

Your Task: Identify the 3 critical bugs in the implementation below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class GPTBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x shape: (batch, seq_len, n_embd)
        B, T, C = x.shape
        
        # Self-Attention Step
        x = self.ln1(x)
        attn_output, _ = self.attn(x, x, x, attn_mask=self.mask[:T, :T])
        x = x + attn_output
        
        # Feed-Forward Step
        x = self.ln2(x)
        x = x + self.mlp(x)
        
        return x

# Simple Training Loop Setup
model = GPTBlock(n_embd=32, n_head=4, block_size=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Dummy data: (Batch, Seq_Len, Dim)
x = torch.randn(2, 8, 32)
targets = torch.randint(0, 32, (2, 8)) # Simplified target labels















"""
Answer:

# Bug 1 is hidden in how this mask is generated or used

# Bug 2 relates to the residual connection or LayerNorm placement

# Bug 3 is a subtle training/inference logic error
"""