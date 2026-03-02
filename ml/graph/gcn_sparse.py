import torch
from torch import nn

class GCNMP(nn.Module):
    def __init__():
        super().__init__()
        


def main():
    num_nodes = 100
    node_feat_dim = 64

    x = torch.rand(num_nodes, node_feat_dim)
    print (x[0])
    edge_index = torch.randint(0, 100, (2, 500))

    hidden_dim = 128

    mp_layer = GCNMP(in_dim=node_feat_dim, hidden_dim=hidden_dim)
    x_hid = mp_layer(x, edge_index)
    print (x.shape)
    assert x_hid.shape == (num_nodes, hidden_dim)
    print ("x_hid.shape: ", x_hid.shape)
    print (x_hid[0])
