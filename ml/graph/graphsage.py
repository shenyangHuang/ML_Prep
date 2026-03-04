import torch
import torch.nn as nn
import torch.nn.functional as F

class SageLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(SageLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Fully connected layer for the concatenated self and neighbor features
        self.projection = nn.Linear(in_features * 2, out_features, bias=bias)

    def forward(self, x, adj):
        """
        x: Node features [N, in_features]
        adj: Adjacency matrix (normalized or sparse) [N, N]
        """
        # 1. Aggregate: Simple mean of neighbors
        # For a standard adj matrix, this is (D^-1)A * X
        neighbor_feats = torch.mm(adj, x)
        
        # 2. Combine: Concatenate self features with aggregated neighbor features
        combined = torch.cat([x, neighbor_feats], dim=1)
        
        # 3. Project & Non-linearity
        out = self.projection(combined)
        return F.relu(out)
    




import torch
import torch.nn as nn
import torch.nn.functional as F

class SageLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(SageLayer, self).__init__()
        # Concatenation of [self, neighbors] means input is 2x in_features
        self.lin = nn.Linear(in_features * 2, out_features)

    def forward(self, x, edge_index):
        """
        x: [N, in_features]
        edge_index: [2, E] where [0, :] is source and [1, :] is target
        """
        row, col = edge_index
        num_nodes = x.size(0)

        # 1. MESSAGE: Get the features of all source nodes (neighbors)
        # msg shape: [E, in_features]
        msg = x[row]

        # 2. AGGREGATE: Collect messages at target nodes (col)
        # Using 'mean' as the aggregator. 
        # include_self=False because we handle the self-node in the CONCAT step.
        neigh_feat = torch.zeros_like(x)
        neigh_feat.scatter_reduce_(0, col.unsqueeze(1).expand_as(msg), msg, 
                                   reduce='mean', include_self=False)

        # 3. COMBINE: [x_v, x_neigh]
        combined = torch.cat([x, neigh_feat], dim=1)
        
        # 4. UPDATE: Linear layer + Activation + L2 Norm
        out = self.lin(combined)
        out = F.relu(out)
        return F.normalize(out, p=2, dim=1)
    



import torch
import torch.nn as nn
import torch.nn.functional as F

def test_graphsage_implementations():
    # 1. Setup Data
    num_nodes = 3
    in_channels = 4
    out_channels = 2
    
    # Feature matrix (X)
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.0], # Node 0
        [0.0, 1.0, 0.0, 0.0], # Node 1
        [0.0, 0.0, 1.0, 0.0]  # Node 2
    ], dtype=torch.float)

    # Edge Index (COO format: Source -> Target)
    # 1->0, 2->0 (Neighbors of 0 are 1 and 2)
    # 0->2       (Neighbor of 2 is 0)
    edge_index = torch.tensor([
        [1, 2, 0], # Source
        [0, 0, 2]  # Target
    ], dtype=torch.long)

    # Dense Adjacency Matrix (Normalized for Mean Aggregation)
    # Row i represents neighbors of node i
    adj = torch.tensor([
        [0.0, 0.5, 0.5], # Node 0 gets 50% from Node 1, 50% from Node 2
        [0.0, 0.0, 0.0], # Node 1 has no neighbors
        [1.0, 0.0, 0.0]  # Node 2 gets 100% from Node 0
    ], dtype=torch.float)

    # 2. Initialize layers with the SAME weights for comparison
    dense_layer = SageLayerDense(in_channels, out_channels)
    sparse_layer = SageLayerEdgeIndex(in_channels, out_channels)
    
    # Copy weights to ensure parity
    sparse_layer.lin.weight.data = dense_layer.projection.weight.data.clone()
    sparse_layer.lin.bias.data = dense_layer.projection.bias.data.clone()

    # 3. Run Forward Passes
    out_dense = dense_layer(x, adj)
    out_sparse = sparse_layer(x, edge_index)

    # 4. Assertions
    print("--- Test Results ---")
    print(f"Dense Output:\n{out_dense}")
    print(f"Sparse Output:\n{out_sparse}")
    
    # Check if they are nearly equal (floating point tolerance)
    is_equal = torch.allclose(out_dense, out_sparse, atol=1e-6)
    print(f"\nMatch Found: {is_equal}")
    
    if is_equal:
        print("✅ Success: Both implementations yield identical embeddings.")
    else:
        print("❌ Failure: Implementations mismatched.")

# --- Required Classes for the Test ---

class SageLayerDense(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.projection = nn.Linear(in_f * 2, out_f)
    def forward(self, x, adj):
        neigh = torch.mm(adj, x)
        return F.normalize(F.relu(self.projection(torch.cat([x, neigh], dim=1))), p=2, dim=1)

class SageLayerEdgeIndex(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.lin = nn.Linear(in_f * 2, out_f)
    def forward(self, x, edge_index):
        row, col = edge_index
        msg = x[row]
        neigh = torch.zeros_like(x)
        # We use scatter_reduce for 'mean'
        neigh.scatter_reduce_(0, col.unsqueeze(1).expand_as(msg), msg, reduce='mean', include_self=False)
        return F.normalize(F.relu(self.lin(torch.cat([x, neigh], dim=1))), p=2, dim=1)

if __name__ == "__main__":
    test_graphsage_implementations()