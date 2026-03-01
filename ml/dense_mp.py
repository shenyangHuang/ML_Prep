"""
dense message passing

H = A W H = A H W
"""


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
      
    def forward(self, x, adj, add_self_loop=False):
        if add_self_loop:
            self_edges = torch.eye(adj.shape[0])
            adj = adj + self_edges
        x = self.linear(x)
        x = torch.matmul(adj, x)  # Ax
        x = self.activation(x)
        return x
    

# Example Usage:
# 3 nodes, 2 features each
x = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
in_dim = 2

# Adjacency matrix (with self-loops to include the node itself)
adj = torch.tensor([[1.0, 1.0, 0.0], 
                    [1.0, 1.0, 1.0], 
                    [0.0, 1.0, 1.0]])

layer = SimpleGNNLayer(2, 4)
output = layer(x, adj)

print("Output shape:", output.shape)
print(output)