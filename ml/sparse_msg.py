import torch 
from torch import nn


class SparseMP(torch.nn.Module):
    def __init__(self,
                 in_dim: int,
                 hidden_dim: int,
                 agg: str = "sum") -> None: 
        super().__init__()     
        self.hid_size = hidden_dim
        self.x_linear = nn.Linear(in_dim, hidden_dim)
        self.msg_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.agg = agg

    def forward(self, x, edge_index):
        x_feat = self.x_linear(x)
        msg = self.msg_func(x_feat, edge_index)
        for i in range(x_feat.shape[0]):
            #! we need to decide on a pooling function here
            if self.agg == "sum":
                msg_set = torch.cat((x_feat[i].view(1,-1), msg[edge_index[0]==i]), dim=0)
                x_feat[i] = torch.sum(msg_set, dim=0)
            else:
                raise NotImplementedError
        return x_feat
    
    def msg_func(self, x_feat, edge_index):
        input_feat = torch.cat((x_feat[edge_index[0]],x_feat[edge_index[1]]), 1)
        msg = self.msg_linear(input_feat) # (num_edge, hidden)
        return msg


def main():
    num_nodes = 100
    node_feat_dim = 64

    x = torch.rand(num_nodes, node_feat_dim)
    edge_index = torch.randint(0, 100, (2, 500))

    hidden_dim = 128

    mp_layer = SparseMP(in_dim=node_feat_dim, hidden_dim=hidden_dim)
    x_hid = mp_layer(x, edge_index)
    assert x_hid.shape == (num_nodes, hidden_dim)


if __name__ == "__main__":
    main()