import torch
from torch.utils.data import Dataset, DataLoader


"""
The core of autoregressive pre-training is shifting the input by one to create the labels.
"""
class GPTDataset(Dataset):
    def __init__(self, txt_data, block_size):
        # txt_data: a long tensor of token IDs
        self.txt_data = txt_data # long stream [T]
        self.block_size = block_size # size of broken down into blocks, seq_len

    def __len__(self):
        return len(self.txt_data) - self.block_size

    def __getitem__(self, idx):
        """
        here we are essentially shifting off by 1
        """
        # Input: tokens from [idx] to [idx + block_size]
        # Target: tokens from [idx + 1] to [idx + block_size + 1]
        x = self.txt_data[idx : idx + self.block_size]
        y = self.txt_data[idx + 1 : idx + self.block_size + 1]  #shift by 1
        return x, y
    

"""
The Training Loop
The key is handling the 3D tensor output from the model to fit the 2D requirement of the Cross-Entropy loss.

Input IDs: (Batch, Seq_Len) — Integers (0 to V-1)

Embeddings: (Batch, Seq_Len, Hidden_Dim) — Continuous vectors

Transformer Blocks: (Batch, Seq_Len, Hidden_Dim) — Stay the same size

LM Head (The Final Linear Layer): Maps Hidden_Dim $\rightarrow$ Vocab_Size.

Output Logits: (Batch, Seq_Len, Vocab_Size) — Raw scores for every word at every position.

"""

import torch.nn.functional as F
def train_step(model, optimizer, x, y):
    optimizer.zero_grad()
    
    # logits shape: (Batch, Seq_Len, Vocab_Size)
    logits = model(x) 
    
    # Flatten Batch and Seq_Len for CrossEntropy
    # B, T, C -> (B*T), C
    B, T, C = logits.shape
    logits_flattened = logits.view(B * T, C)  # here we are predicting the next token for every single word because it is shifted by 1 for each word
    targets_flattened = y.view(B * T) 
    
    loss = F.cross_entropy(logits_flattened, targets_flattened)
    loss.backward()
    optimizer.step()
    
    return loss.item()

"""
complete training loop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- 1. Dataset Logic (The Shift) ---
class GPTDataset(Dataset):
    def __init__(self, data_as_tensor, block_size):
        self.data = data_as_tensor
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # x is the sequence, y is the sequence shifted by 1
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

# --- 2. Minimal GPT Model (with Weight Tying) ---
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """ The 'MLP' block: Linear -> GELU -> Linear """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(), # GPT uses GELU instead of ReLU
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ One Transformer Block: Communication (Attention) + Computation (FFN) """
    def __init__(self, n_embd, n_head):
        super().__init__()
        # Every block has its own LayerNorms
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head) # Implementation from before
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        # Pre-Norm with Residual Connections
        # x + Attention(LN(x))
        x = x + self.attn(self.ln1(x))
        # x + FFN(LN(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size):
        super().__init__()
        self.block_size = block_size
        
        # 1. Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # 2. Transformer Backbone (List of Blocks)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        
        # 3. Final LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # 4. LM Head (tied to embeddings)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight # weight typing to save space

    def forward(self, idx):
        B, T = idx.shape # (B,T)
        
        # Create position indices [0, 1, 2, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        
        # Combine Token + Position Embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(pos) # (T, C)
        x = tok_emb + pos_emb # Broadcasting (B, T, C)
        
        # Pass through Transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Map to vocabulary scores
        logits = self.lm_head(x) # (B, T, V)
        return logits

# --- 3. The Complete Training Loop ---
def train_model():
    # Hyperparameters
    vocab_size = 50257
    block_size = 128
    batch_size = 32
    n_embd = 768
    learning_rate = 3e-4

    # Setup dummy data (integers representing tokens)
    dummy_data = torch.randint(0, vocab_size, (10000,))
    dataset = GPTDataset(dummy_data, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MiniGPT(vocab_size, n_embd)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(1):
        for x, y in loader:
            # x: (B, T), y: (B, T)
            optimizer.zero_grad()

            # 1. Forward pass
            logits = model(x) # Shape: (B, T, V)

            # 2. Reshape for CrossEntropy
            # F.cross_entropy expects (N, C) where C is the number of classes
            B, T, V = logits.shape
            logits_flattened = logits.view(B * T, V)
            targets_flattened = y.view(B * T)

            # 3. Calculate Loss
            # ignore_index could be used if we had padding
            loss = F.cross_entropy(logits_flattened, targets_flattened)

            # 4. Backward pass
            loss.backward()
            
            # 5. Gradient Clipping (Standard GPT practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")
            break # Just showing one step for demonstration

if __name__ == "__main__":
    train_model()





"""
In a coding interview, the goal is to show you understand how to parallelize attention. 
You shouldn't write a for loop over the number of heads; instead, use 4D tensor manipulation to process all heads simultaneously.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Key, Query, Value projections combined into one linear layer
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        B, T, C = x.size() # Batch, Sequence Length, Embedding Dim

        # 1. Project to Q, K, V and split heads
        q, k, v = self.c_attn(x).split(C, dim=2)
        
        # Reshape to (B, n_head, T, head_dim)
        # Transpose is crucial to move head dim to position 1 for batch matrix mult
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        # (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        
        # 3. Apply Causal Mask (Lower Triangular)
        # This prevents the model from looking at "future" tokens
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        
        # 4. Weighted sum of Values
        # (B, nh, T, T) @ (B, nh, T, hs) -> (B, nh, T, hs)
        y = att @ v 
        
        # 5. Re-assemble (Concat heads)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)