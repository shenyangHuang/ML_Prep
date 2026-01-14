import torch
torch.manual_seed(0)

"""
https://chatgpt.com/share/e/6965a805-d950-8009-b94a-d30279729fc8
"""

# dims
N, D, H, C = 64, 2, 16, 1
X = torch.randn(N, D)                      # X
Y = torch.randn(N, C)                      # Y (targets)

# params (no autograd)
W1 = torch.randn(D, H) * 0.1               # W^{(1)}
b1 = torch.zeros(H)                        # b^{(1)}
W2 = torch.randn(H, C) * 0.1               # W^{(2)}
b2 = torch.zeros(C)                        # b^{(2)}

print ("finished initializing vectors and weights")
print("W1 before backprop:")
print(W1)

lr = 1e-2
# --- forward (maps to Z^{(1)}, A^{(1)}, \hat{Y}) ---
Z1 = X @ W1 + b1            # Z^{(1)}   (broadcast b1 across batch)
A1 = torch.maximum(Z1, torch.tensor(0.0))  # A^{(1)} = ReLU(Z1)
Y_hat = A1 @ W2 + b2        # \hat{Y}

# --- loss ---
E = Y_hat - Y               # E = \hat{Y} - Y
loss = (E**2).mean()        # L = (1/N) * sum squares

# --- backward (maps to D^{(2)}, grad W2/b2, D^{(1)}, grad W1/b1) ---
D2 = (2.0 / N) * E          # D^{(2)} = dL/dY_hat, N from the mean and 2 from the square

grad_W2 = A1.T @ D2         # dL/dW^{(2)} = A^{(1)T} @ D^{(2)}
grad_b2 = D2.sum(dim=0)     # dL/db^{(2)} = sum over batch

D1_a = D2 @ W2.T            # D^{(1)}_A = D^{(2)} @ W^{(2)T}
relu_mask = (Z1 > 0).float()# mask = I(Z^{(1)} > 0)
D1 = D1_a * relu_mask       # D^{(1)} = D1_a ⊙ mask

grad_W1 = X.T @ D1          # dL/dW^{(1)} = X^T @ D^{(1)}
grad_b1 = D1.sum(dim=0)     # dL/db^{(1)} = sum over batch

# --- param update (gradient descent) ---
W2 -= lr * grad_W2
b2 -= lr * grad_b2
W1 -= lr * grad_W1
b1 -= lr * grad_b1

print("W1 after backprop:")
print(W1)

