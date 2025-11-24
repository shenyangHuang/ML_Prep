# based on the tutorial from
# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial7/GNN_overview.html
from flax import linen as nn
import jax
import jax.numpy as jnp


edge_index = jnp.array([
        [0, 1, 0, 2, 1],  # sources
        [1, 0, 2, 0, 3]   # targets
    ], dtype=int)

feat_size = 100

key = jax.random.PRNGKey(0)
node_feats = jax.random.normal(key, (4, feat_size))


class GCNLayer(nn.Module):
    c_out : int

    def __init__(self, 
                 in_size: int, 
                 hid_size: int):
        self.hid_size = hid_size
        self.nbr_w = jax.random.normal(key, (in_size,hid_size))
        self.self_w = jax.random.normal(key, (in_size,hid_size))
        self.b = jax.random.normal(key, (hid_size))

    def __call__(self, node_feats, edge_index):
        """
        Inputs:
            node_feats - Array with node features of shape [num_nodes, c_in]
            edge_index - Array with shape [2, num_edges] describing the edges of the graph.
                         edge_index[0,i] = source node of edge i
                         edge_index[1,i] = target node of edge i
        """
        # Num neighbours = number of incoming edges
        h = node_feats
        src = edge_index[0]
        dst = edge_index[1]
        # msg = h.clone()
        # out = jnp.zeros((h.shape[0], self.hid_size))
        # for i in range(len(msg)):
        #     nbr_msg = jnp.sum(h[dst[src == i]], axis=0)
        #     nbr_msg = jnp.tanh( nbr_msg @ self.nbr_w + self.b) 
        #     out.at[i].set(msg[i] @ self.self_w + nbr_msg) #tanh activation
        # h = msg
        # return h

        msg = h
        nbr_features = h[dst]
        nbr_sum = jax.ops.segment_sum(nbr_features, src, num_segments=h.shape[0])
        nbr_msg = jnp.tanh(nbr_sum @ self.nbr_w + self.b)
        self_msg = msg @ self.self_w
        out = self_msg + nbr_msg
        return out
 
in_size = feat_size

gcn = GCNLayer(in_size=feat_size, hid_size=64)
out_h = gcn(node_feats, edge_index)
print (out_h[0])

        

