import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(GraphConvolutionLayer, self).__init__()
        self.dense = nn.Linear(in_channels, out_channels, bias=bias)

        self.apply(weights_init)

    def forward(self, x, adj):
        out = self.dense(x)
        out = torch.mm(adj, out)
        return out


# temp = torch.randn((5, 10))
# adj = torch.randn((5, 5))
# gcn = GraphConvolutionLayer(10, 64)
#
# print(gcn(temp, adj).shape)
