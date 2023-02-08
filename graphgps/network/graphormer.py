import torch
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network

from graphgps.layer.graphormer_layer import GraphormerLayer


@register_network('Graphormer')
class GraphormerModel(torch.nn.Module):
    """Graphormer port to GraphGPS.
    https://arxiv.org/abs/2106.05234
    Ying, C., Cai, T., Luo, S., Zheng, S., Ke, G., He, D., ... & Liu, T. Y.
    Do transformers really perform badly for graph representation? (NeurIPS2021)
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        if not cfg.graphormer.embed_dim == cfg.gnn.dim_inner == dim_in:
            raise ValueError(
                f"The inner and embed dims must match: "
                f"embed_dim={cfg.graphormer.embed_dim} "
                f"dim_inner={cfg.gnn.dim_inner} dim_in={dim_in}"
            )

        layers = []
        for _ in range(cfg.graphormer.num_layers):
            layers.append(GraphormerLayer(
                embed_dim=cfg.graphormer.embed_dim,
                num_heads=cfg.graphormer.num_heads,
                dropout=cfg.graphormer.dropout,
                attention_dropout=cfg.graphormer.attention_dropout,
                mlp_dropout=cfg.graphormer.mlp_dropout
            ))
        self.layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
