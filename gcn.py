# based on the tutorial from
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial7/GNN_overview.html
from flax import linen as nn
import jax

class GCNLayer(nn.Module):
    c_out : int

    @nn.compact
    def __call__(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Array with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(axis=-1, keepdims=True)
        node_feats = nn.Dense(features=self.c_out, name='projection')(node_feats)
        node_feats = jax.lax.batch_matmul(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats

