import torch 
from torch import nn
import torch.nn.functional as F




class SparseMP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, gat: bool=False):
        super().__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()
        if gat:
            self.a = nn.Parameter(torch.empty(size=(2 * hidden_dim, 1)))




    def forward(self, x, edge_index):
        x = self.linear(x)
        msgs = self.msg_func(x, edge_index)
        init_feat = torch.zeros_like(x)
        src, dst = edge_index[0], edge_index[1]
        out = init_feat.index_add_(0, src, msgs)
        out = self.activation(out)
        return out
    

    def msg_func(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        msgs = x[src] + x[dst]
        return msgs
    
    def gat_msg_func(self, x, edge_index):
        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        src, dst = edge_index[0], edge_index[1]
        e = torch.cat([x[src], x[dst]], dim=-1) @ self.a # (E, 1)
        e = F.leaky_relu(e, negative_slope=0.2)


    """
    to check
    """
    def sparse_softmax(self, src, index, num_nodes):
        """
        Numerically stable softmax for sparse indices.
        src: [E], index: [E] (target node indices)
        """
        # Subtract max for stability
        src_max = torch.zeros(num_nodes, device=src.device)
        src_max.index_reduce_(0, index, src, reduce='amax', include_self=False)
        src_stable = src - src_max[index]
        
        exp = src_stable.exp()
        sum_exp = torch.zeros(num_nodes, device=src.device)
        sum_exp.index_add_(0, index, exp)
        
        return exp / (sum_exp[index] + 1e-16)








def main():
    num_nodes = 100
    node_feat_dim = 64

    x = torch.rand(num_nodes, node_feat_dim)
    print (x[0])
    edge_index = torch.randint(0, 100, (2, 500))

    hidden_dim = 128

    mp_layer = SparseMP(in_dim=node_feat_dim, hidden_dim=hidden_dim)
    x_hid = mp_layer(x, edge_index)
    print (x.shape)
    assert x_hid.shape == (num_nodes, hidden_dim)
    print ("x_hid.shape: ", x_hid.shape)
    print (x_hid[0])


if __name__ == "__main__":
    main()






"""
implementation 2025
"""
# class SparseMP(torch.nn.Module):
#     def __init__(self,
#                  in_dim: int,
#                  hidden_dim: int,
#                  agg: str = "sum") -> None: 
#         super().__init__()     
#         self.hid_size = hidden_dim
#         self.x_linear = nn.Linear(in_dim, hidden_dim)
#         self.msg_linear = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.agg = agg

#     def forward(self, x, edge_index):
#         x_feat = self.x_linear(x)
#         msg = self.msg_func(x_feat, edge_index)

#         """
#         #! with a for loop
#         for i in range(x_feat.shape[0]):
#             if self.agg == "sum":
#                 msg_set = torch.cat((x_feat[i].view(1,-1), msg[edge_index[0]==i]), dim=0)
#                 x_feat[i] = torch.sum(msg_set, dim=0)
#             else:
#                 raise NotImplementedError
#         """
#         # Aggregate edge messages per node without a Python loop.
#         if self.agg == "sum":
#             # Matches the old loop semantics:
#             # x_out[i] = x_feat[i] + sum_{e: edge_index[0, e] == i} msg[e]
#             # https://docs.pytorch.org/docs/stable/generated/torch.zeros_like.html
#             # https://docs.pytorch.org/docs/stable/generated/torch.index_add_.html
#             agg_msg = torch.zeros_like(x_feat).index_add_(0, edge_index[0], msg)
#             return x_feat + agg_msg
#         raise NotImplementedError
    
#     def msg_func(self, x_feat, edge_index):

#         #! implement degree normalization
#         node_ids, node_counts = torch.unique(edge_index[0], return_counts=True)
#         deg = torch.zeros(x_feat.shape[0], dtype=int)
#         deg[node_ids.int()] = node_counts + 1
#         deg_norm = deg.sqrt().view(-1,1)

#         input_feat = torch.cat((x_feat[edge_index[0]] * deg_norm[edge_index[0]], x_feat[edge_index[1]] * deg_norm[edge_index[1]]), 1)
#         msg = self.msg_linear(input_feat) # (num_edge, hidden)
#         return msg


# def main():
#     num_nodes = 100
#     node_feat_dim = 64

#     x = torch.rand(num_nodes, node_feat_dim)
#     edge_index = torch.randint(0, 100, (2, 500))

#     hidden_dim = 128

#     mp_layer = SparseMP(in_dim=node_feat_dim, hidden_dim=hidden_dim)
#     x_hid = mp_layer(x, edge_index)
#     print (x.shape)
#     assert x_hid.shape == (num_nodes, hidden_dim)
#     print ("x_hid.shape: ", x_hid.shape)


# if __name__ == "__main__":
#     main()