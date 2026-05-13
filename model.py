import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv,GCNConv
from torch_geometric.utils import k_hop_subgraph, to_undirected, add_self_loops


class GI_GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,  max_iter, eps, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hid_dim))
        for i in range(max_iter-1):
            self.layers.append(GCNConv(hid_dim, hid_dim))

        self.layers.append(GCNConv(hid_dim, out_dim))
        self.eps = eps
        self.dropout = dropout
        self.max_iter = max_iter

    def forward(self, x, edge_index, batch):
        N, Fdim = x.shape
        p = torch.ones(N, device=x.device, dtype=x.dtype)
        for layer_idx in range(self.max_iter):

            for i in range(1):
                x_centered = x - x.mean(dim=1, keepdim=True)
                std = x_centered.std(dim=1, keepdim=True) + 1e-6
                x_normed = x_centered / std
                M = torch.einsum('n,nf,ng->fg', p, x_normed, x_normed)  # (F, F)
                Ap = torch.einsum('nf,fg,ng->n', x_normed, M, x_normed)
                p_new = p * Ap / (torch.dot(p, Ap) + 1e-9)
                p_new = p_new / p_new.sum()
                p = p_new

            p_weight = p * N
            x_weighted = x * p_weight.unsqueeze(1)
            x = self.layers[layer_idx](x_weighted, edge_index)
            x = F.dropout(x, p=0.5, training=self.training)
            if layer_idx != self.max_iter - 1:
                x = F.relu(x)
        return x