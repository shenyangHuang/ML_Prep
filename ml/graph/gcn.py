import torch
from torch import nn

class GCNDense(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.layer = nn.Linear(in_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x, adj):
        # degree normalization for gcn, with sqrt
        dig = torch.sum(adj, dim=1) + 1
        d = torch.diag(dig ** (-0.5))  #empty still is 0
        adj = adj + torch.eye(adj.shape[0])

        norm_adj = d @ adj.float() @ d 

        x = self.layer(x)
        x = torch.matmul(norm_adj, x)  # Ax
        x = self.activation(x)
        return x


class GCNSparse(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.layer = nn.Linear(2*in_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x, edge_index):
        src, dst = edge_index[0], edge_index[1]
        msg = self.msg_func(x, edge_index)
        out = torch.zeros(x.shape[0], self.hidden_dim)
        out.index_add(0,src,msg)
        out = self.activation(out)

    def msg_func(self, x, edge_index):
        x = self.layer(x)
        src, dst = edge_index[0], edge_index[1]
        nodes, deg = torch.unique(src, return_count=True)
        # there are nodes with no degree as well
        msg = x[src] + x[dst]
    





# class GCNSparse(nn.Module):
#     def __init__(self, in_dim, hidden_dim):
#         super().__init__()
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.layer = nn.Linear(2*in_dim, hidden_dim)
#         self.activation = nn.ReLU()
    
#     def forward(self, x, edge_index):
#         src, dst = edge_index[0], edge_index[1]

#         msg = self.msg_func(x, edge_index)
#         embed = torch.zeros(x.shape[0], msg.shape[1])
#         embed = embed.index_add(0, src, msg)
#         out = self.activation(embed)
#         return out



#     def msg_func(self, x, edge_index):
#         src, dst = edge_index[0], edge_index[1]
#         node_idx, freq = torch.unique(src, return_counts=True)
#         deg = torch.zeros(x.shape[0]).long()
#         deg[node_idx.int()] = freq # adding self edge so there is no 0 degree node
#         deg = deg + 1
#         deg_norm = deg.sqrt().view(-1,1)
#         input_feat = torch.cat((x[edge_index[0]] * deg_norm[edge_index[0]], x[edge_index[1]] * deg_norm[edge_index[1]]), 1)
#         msg = self.layer(input_feat)
#         return msg

    


        




def main():
    #! sparse message passing
    num_nodes = 100
    node_feat_dim = 64
    x = torch.rand(num_nodes, node_feat_dim)
    edge_index = torch.randint(0, 100, (2, 500))


    #! adj
    adj = torch.zeros((num_nodes, num_nodes))
    adj[edge_index[0], edge_index[1]] = 1.0

    # model definition
    hidden_dim = 128

    #! Dense
    mp_layer = GCNDense(in_dim=node_feat_dim, hidden_dim=hidden_dim)
    x_hid = mp_layer(x, adj)
    print (x.shape)
    assert x_hid.shape == (num_nodes, hidden_dim)
    print ("x_hid.shape: ", x_hid.shape)
    print (x_hid[0])
    
    
    # #! Sparse
    # mp_layer = GCNSparse(in_dim=node_feat_dim, hidden_dim=hidden_dim)
    # x_hid = mp_layer(x, edge_index)
    # print (x.shape)
    # assert x_hid.shape == (num_nodes, hidden_dim)
    # print ("x_hid.shape: ", x_hid.shape)
    # print (x_hid[0])








    

    # mp_layer = GCNMP(in_dim=node_feat_dim, hidden_dim=hidden_dim)
    # x_hid = mp_layer(x, edge_index)
    # print (x.shape)
    # assert x_hid.shape == (num_nodes, hidden_dim)
    # print ("x_hid.shape: ", x_hid.shape)
    # print (x_hid[0])



if __name__ == "__main__":
    main()