import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphormerAttention(nn.Module):
    def __init__(self, dim, num_heads, n_distances, n_edge_types):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # 1. Spatial Encoding: Bias based on shortest path distance
        self.spatial_pos_bias = nn.Embedding(n_distances, num_heads)
        
        # 2. Edge Encoding: Bias based on edge features along the shortest path
        # In the original paper, this is an average of edge embeddings
        self.edge_bias = nn.Embedding(n_edge_types, num_heads)
        
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, dist_matrix, edge_features=None):
        """
        x: [N, dim]
        dist_matrix: [N, N] (Shortest Path Distance)
        edge_features: [N, N] (Type of edge if it exists)
        """
        N, C = x.shape
        q = self.q_proj(x).view(N, self.num_heads, self.d_k).transpose(0, 1)
        k = self.k_proj(x).view(N, self.num_heads, self.d_k).transpose(0, 1)
        v = self.v_proj(x).view(N, self.num_heads, self.d_k).transpose(0, 1)

        # Base Attention: [H, N, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Add Spatial Encoding Bias
        # dist_matrix: [N, N] -> bias: [N, N, H] -> [H, N, N]
        b_phi = self.spatial_pos_bias(dist_matrix).permute(2, 0, 1)
        attn = attn + b_phi

        # Add Edge Encoding Bias (Simplified for direct neighbors)
        if edge_features is not None:
            e_phi = self.edge_bias(edge_features).permute(2, 0, 1)
            attn = attn + e_phi

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(0, 1).reshape(N, C)
        return self.out_proj(out)
    



class Graphormer(nn.Module):
    def __init__(self, n_layers, dim, num_heads, max_degree, n_distances, n_edge_types):
        super().__init__()
        # Centrality Encoding: Learnable embeddings for In-Degree and Out-Degree
        self.in_degree_encoder = nn.Embedding(max_degree, dim)
        self.out_degree_encoder = nn.Embedding(max_degree, dim)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': GraphormerAttention(dim, num_heads, n_distances, n_edge_types),
                'norm1': nn.LayerNorm(dim),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
                'norm2': nn.LayerNorm(dim)
            }) for _ in range(n_layers)
        ])

    def forward(self, x, edge_index, dist_matrix, edge_attr_matrix=None):
        N = x.size(0)
        
        # 1. Compute Degrees for Centrality Encoding
        # In an interview, explain that degree captures node importance
        in_degree = torch.zeros(N, dtype=torch.long, device=x.device)
        out_degree = torch.zeros(N, dtype=torch.long, device=x.device)
        in_degree.index_add_(0, edge_index[1], torch.ones(edge_index.size(1), device=x.device).long())
        out_degree.index_add_(0, edge_index[0], torch.ones(edge_index.size(1), device=x.device).long())

        # 2. Initial Embedding = Node Features + Centrality
        h = x + self.in_degree_encoder(in_degree) + self.out_degree_encoder(out_degree)

        # 3. Transformer Blocks
        for layer in self.layers:
            # Attention + Residual
            h = h + layer['attn'](layer['norm1'](h), dist_matrix, edge_attr_matrix)
            # FFN + Residual
            h = h + layer['ffn'](layer['norm2'](h))
            
        return h