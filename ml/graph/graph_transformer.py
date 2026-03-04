import torch
import torch.nn as nn
import torch.nn.functional as F


import torch

def compute_laplacian_pe(edge_index, num_nodes, k=8):
    """
    Computes Laplacian PE without PyG dependencies.
    
    Args:
        edge_index: torch.Tensor [2, E] (COO format)
        num_nodes: int
        k: number of eigenvectors to return
    """
    # 1. Construct Adjacency Matrix (A)
    # We use a dense matrix here for the eigendecomposition
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1.0
    
    # 2. Compute Degree Matrix (D)
    deg = torch.sum(adj, dim=1)
    
    # 3. Compute Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
    # Handle isolated nodes (degree 0) to avoid division by zero
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    
    # L = I - D_inv_sqrt @ A @ D_inv_sqrt
    identity = torch.eye(num_nodes)
    L = identity - torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
    
    # 4. Eigendecomposition
    # eigh is for symmetric matrices (Laplacian is symmetric)
    # Returns eigenvalues in ascending order
    eigvals, eigvecs = torch.linalg.eigh(L)
    
    # 5. Extract k-smallest non-trivial eigenvectors
    # We skip the 1st eigenvector (index 0) because it is constant/trivial
    # for a connected graph (eigenvalue is 0).
    pe = eigvecs[:, 1:k+1]
    
    # Edge case: If the graph is small or disconnected, pad with zeros if needed
    if pe.shape[1] < k:
        padding = torch.zeros((num_nodes, k - pe.shape[1]))
        pe = torch.cat([pe, padding], dim=1)
        
    return pe

class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, max_path_distance=5):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = out_dim // num_heads
        
        # Standard Attention Projections
        self.q = nn.Linear(in_dim, out_dim)
        self.k = nn.Linear(in_dim, out_dim)
        self.v = nn.Linear(in_dim, out_dim)
        
        # Spatial Bias: A learnable parameter for each possible distance
        # This tells the model how much to "trust" a node k-steps away
        self.spatial_bias = nn.Embedding(max_path_distance + 1, num_heads)
        
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, dist_matrix):
        """
        x: [N, in_dim] - Node features
        dist_matrix: [N, N] - Shortest path distances between all nodes
        """
        N = x.size(0)
        
        # 1. Project to Q, K, V and split into heads
        # Shape: [num_heads, N, d_k]
        q = self.q(x).view(N, self.num_heads, self.d_k).transpose(0, 1)
        k = self.k(x).view(N, self.num_heads, self.d_k).transpose(0, 1)
        v = self.v(x).view(N, self.num_heads, self.d_k).transpose(0, 1)

        # 2. Scaled Dot-Product Attention
        # Shape: [num_heads, N, N]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # 3. Inject Graph Structure (The "Secret Sauce")
        # Map distances to learnable biases: [N, N, num_heads] -> [num_heads, N, N]
        bias = self.spatial_bias(dist_matrix).permute(2, 0, 1)
        attn_scores = attn_scores + bias

        # 4. Softmax & Aggregate
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, v) # [num_heads, N, d_k]
        
        # 5. Concatenate heads and project
        out = out.transpose(0, 1).contiguous().view(N, -1)
        out = self.out_proj(out)
        
        return self.norm(out + x) # Residual connection
    






import torch
import pytest

# Assuming the GraphTransformerLayer and compute_laplacian_pe are defined above

def test_graph_transformer_permutation_invariance():
    """
    Core Test: If we shuffle the nodes, the output features should 
    be the same (once un-shuffled).
    """
    num_nodes = 4
    in_dim = 16
    out_dim = 16
    
    # 1. Create a simple line graph: 0-1-2-3
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], 
                               [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    x = torch.randn(num_nodes, in_dim)
    
    # Simple Distance Matrix for 0-1-2-3
    # (In a real scenario, use BFS/Floyd-Warshall to compute this)
    dist_matrix = torch.tensor([
        [0, 1, 2, 3],
        [1, 0, 1, 2],
        [2, 1, 0, 1],
        [3, 2, 1, 0]
    ])

    model = GraphTransformerLayer(in_dim, out_dim, num_heads=2)
    model.eval()

    # Original Forward Pass
    with torch.no_grad():
        out1 = model(x, dist_matrix)

    # 2. Shuffle nodes: Swap index 0 and 3
    perm = torch.tensor([3, 1, 2, 0])
    x_shuffled = x[perm]
    dist_shuffled = dist_matrix[perm][:, perm]

    with torch.no_grad():
        out2 = model(x_shuffled, dist_shuffled)

    # 3. Un-shuffle the output
    # The output at index 0 of out1 should match index 3 of out2
    inv_perm = torch.argsort(perm)
    out2_unshuffled = out2[inv_perm]

    # Assert equality with a small tolerance for floating point
    assert torch.allclose(out1, out2_unshuffled, atol=1e-5), "Model is NOT permutation invariant!"

def test_graph_transformer_output_shape():
    """
    Check if dimensions are preserved correctly.
    """
    N, D_in, D_out = 5, 8, 16
    x = torch.randn(N, D_in)
    dist = torch.zeros((N, N)).long() # Identity distance
    
    model = GraphTransformerLayer(D_in, D_out, num_heads=4)
    output = model(x, dist)
    
    assert output.shape == (N, D_out), f"Expected {(N, D_out)}, got {output.shape}"

def test_laplacian_pe_orthogonality():
    """
    Check if the eigenvectors computed are actually orthogonal (V^T @ V = I).
    """
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    num_nodes = 3
    k = 2
    
    pe = compute_laplacian_pe(edge_index, num_nodes, k=k)
    
    # Orthogonality check: pe.T @ pe should be Identity
    res = torch.mm(pe.t(), pe)
    expected = torch.eye(k)
    
    assert torch.allclose(res, expected, atol=1e-5), "Laplacian PE vectors are not orthogonal"