Preparing for an ML coding interview on GPT (Generative Pre-trained Transformer) requires understanding the **autoregressive objective**: predicting the next token given a sequence of previous tokens.

Below is a concise PyTorch implementation and a breakdown of the "gotchas" that interviewers love to test.

## ---

**1\. Minimal Implementation (PyTorch)**

In an interview, you aren't expected to write 1,000 lines. Focus on the **Dataset** logic and the **Training Loop**, as these demonstrate your understanding of the autoregressive nature.

### **The Dataset: Sliding Window**

The core of autoregressive pre-training is shifting the input by one to create the labels.

Python

import torch  
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):  
    def \_\_init\_\_(self, txt\_data, block\_size):  
        \# txt\_data: a long tensor of token IDs  
        self.txt\_data \= txt\_data  
        self.block\_size \= block\_size

    def \_\_len\_\_(self):  
        return len(self.txt\_data) \- self.block\_size

    def \_\_getitem\_\_(self, idx):  
        \# Input: tokens from \[idx\] to \[idx \+ block\_size\]  
        \# Target: tokens from \[idx \+ 1\] to \[idx \+ block\_size \+ 1\]  
        x \= self.txt\_data\[idx : idx \+ self.block\_size\]  
        y \= self.txt\_data\[idx \+ 1 : idx \+ self.block\_size \+ 1\]  
        return x, y

### **The Training Loop**

The key is handling the 3D tensor output from the model to fit the 2D requirement of the Cross-Entropy loss.

Python

import torch.nn.functional as F

def train\_step(model, optimizer, x, y):  
    optimizer.zero\_grad()  
      
    \# logits shape: (Batch, Seq\_Len, Vocab\_Size)  
    logits \= model(x)   
      
    \# Flatten Batch and Seq\_Len for CrossEntropy  
    \# B, T, C \-\> (B\*T), C  
    B, T, C \= logits.shape  
    logits\_flattened \= logits.view(B \* T, C)  
    targets\_flattened \= y.view(B \* T)  
      
    loss \= F.cross\_entropy(logits\_flattened, targets\_flattened)  
    loss.backward()  
    optimizer.step()  
      
    return loss.item()

## ---

**2\. Common Implementation Bugs**

When an interviewer asks "What could go wrong?", they are checking your "debugging intuition."

### **A. The "Off-by-One" Label Shift**

* **The Bug:** Forgetting to shift the targets. If x \== y, the model learns the identity function (copying the input) rather than predicting the *next* token.  
* **Correction:** Ensure y \= tokens\[1:\] and x \= tokens\[:-1\].

### **B. Causal Masking (The "Cheating" Bug)**

* **The Bug:** In the self-attention layer, if you don't apply a **triangular mask** (causal mask), the model can "see" the future tokens in the sequence during the forward pass.  
* **Symptom:** Training loss drops to nearly zero instantly, but the model produces gibberish during inference because it never learned to predict without "peeking."

### **C. Cross-Entropy Shape Mismatch**

* **The Bug:** Passing the raw (B, T, C) logits into F.cross\_entropy.  
* **Correction:** PyTorch's cross\_entropy expects the class dimension to be second, or a flattened 2D input. You must use .view(-1, C) on logits and .view(-1) on targets.

### **D. Weight Tying Errors**

* **The Bug:** GPT models often "tie" (share) the weights of the input embedding and the final output linear layer. If you forget to initialize them to the same memory pointer, the model has \~30% more parameters than intended and trains slower.  
* **Code check:** self.lm\_head.weight \= self.transformer.wte.weight

## ---

**3\. Interview "Pro-Tips"**

1. **Mention KV Caching:** If they ask about **inference**, mention that we don't re-calculate the entire sequence for every new token; we cache the Key and Value tensors.  
2. **Gradient Clipping:** Pre-training is unstable. Always mention that you'd use torch.nn.utils.clip\_grad\_norm\_ to prevent exploding gradients.  
3. **Position Embeddings:** Remind them that Transformers are permutation-invariant. Without adding **Positional Encodings**, the model treats "The dog bit the man" and "The man bit the dog" as identical.

**Would you like me to walk through the Multi-Head Attention implementation specifically, or perhaps the logic for KV caching?**