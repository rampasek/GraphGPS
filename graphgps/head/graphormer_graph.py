import torch

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('graphormer_graph')
class GraphormerHead(torch.nn.Module):
    """
    Graphormer prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        print(f"Initializing {cfg.model.graph_pooling} pooling function")
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]

        self.ln = torch.nn.LayerNorm(dim_in)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out)
        )

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch):
        x = self.ln(batch.x)
        graph_emb = self.pooling_fun(x, batch.batch)
        graph_emb = self.layers(graph_emb)
        batch.graph_feature = graph_emb
        pred, label = self._apply_index(batch)
        return pred, label
