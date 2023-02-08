import torch
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head


@register_head('infer_links')
class InferLinksHead(torch.nn.Module):
    """
    InferLinks prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        if cfg.dataset.infer_link_label == "edge":
            dim_out = 2
        else:
            raise ValueError(f"Infer-link task {cfg.dataset.infer_link_label} not available.")

        self.predictor = torch.nn.Linear(1, dim_out)

    def forward(self, batch):
        x = batch.x[batch.complete_edge_index]
        x = (x[0] * x[1]).sum(1)
        y = self.predictor(x.unsqueeze(1))
        return y, batch.y
