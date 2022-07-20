import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.bigbird_layer import BigBirdModel as BackboneBigBird


@register_network('BigBird')
class BigBird(torch.nn.Module):
    """BigBird without edge features.
    This model disregards edge features and runs a linear transformer over a set of node features only.
    BirBird applies random sparse attention to the input sequence - the longer the sequence the closer it is to O(N)
    https://arxiv.org/abs/2007.14062
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

        # Copy main Transformer hyperparams to the BigBird config.
        cfg.gt.bigbird.layers = cfg.gt.layers
        cfg.gt.bigbird.n_heads = cfg.gt.n_heads
        cfg.gt.bigbird.dim_hidden = cfg.gt.dim_hidden
        cfg.gt.bigbird.dropout = cfg.gt.dropout
        self.trf = BackboneBigBird(
            config=cfg.gt.bigbird,
        )

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
