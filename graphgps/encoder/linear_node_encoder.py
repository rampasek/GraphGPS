import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('LinearNode')
class LinearNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        
        self.encoder = torch.nn.Linear(cfg.share.dim_in, emb_dim)

    def forward(self, batch):
        batch.x = self.encoder(batch.x)
        return batch
