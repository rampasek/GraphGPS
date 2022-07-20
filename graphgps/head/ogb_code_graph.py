import torch.nn as nn

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('ogb_code_graph')
class OGBCodeGraphHead(nn.Module):
    """
    Sequence prediction head for ogbg-code2 graph-level prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): IGNORED, kept for GraphGym framework compatibility
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out, L=1):
        super().__init__()
        self.pooling_fun = register.pooling_dict[cfg.model.graph_pooling]
        self.L = L
        num_vocab = 5002
        self.max_seq_len = 5

        if self.L != 1:
            raise ValueError(f"Multilayer prediction heads are not supported.")

        self.graph_pred_linear_list = nn.ModuleList()
        for i in range(self.max_seq_len):
            self.graph_pred_linear_list.append(nn.Linear(dim_in, num_vocab))

    def _apply_index(self, batch):
        return batch.pred_list, {'y_arr': batch.y_arr, 'y': batch.y}

    def forward(self, batch):
        graph_emb = self.pooling_fun(batch.x, batch.batch)

        pred_list = []
        for i in range(self.max_seq_len):
            pred_list.append(self.graph_pred_linear_list[i](graph_emb))
        batch.pred_list = pred_list

        pred, label = self._apply_index(batch)
        return pred, label
