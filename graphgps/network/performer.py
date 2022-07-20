import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.performer_layer import Performer as BackbonePerformer


@register_network('Performer')
class Performer(torch.nn.Module):
    """Performer without edge features.
    This model disregards edge features and runs a linear transformer over a set of node features only.
    https://arxiv.org/abs/2009.14794
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gt.dim_hidden == cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."

        self.trf = BackbonePerformer(
            dim=cfg.gt.dim_hidden,
            depth=cfg.gt.layers,
            heads=cfg.gt.n_heads,
            dim_head=cfg.gt.dim_hidden // cfg.gt.n_heads
        )

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
