import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn import GraphConvolutionLayer


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.graph1 = GraphConvolutionLayer(16, 64)
        self.graph2 = GraphConvolutionLayer(64, 128)
        self.graph3 = GraphConvolutionLayer(128, 10)

    def forward(self, x, adj):
        out = F.relu(self.graph1(x, adj))
        out = F.relu(self.graph2(out, adj))
        out = self.graph3(out, adj)
        return out


temp = torch.randn((5, 16))
adj = torch.randn((5, 5))
model = Network()

print(model(temp, adj).shape)
