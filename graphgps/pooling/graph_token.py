from torch_geometric.graphgym.register import register_pooling
from torch_geometric.utils import to_dense_batch


@register_pooling('graph_token')
def graph_token_pooling(x, batch, *args):
    """Extracts the graph token from a batch to perform graph-level prediction.
    Typically used together with Graphormer when GraphormerEncoder is used and
    the global graph token is used: `cfg.graphormer.use_graph_token == True`.
    """
    x, _ = to_dense_batch(x, batch)
    return x[:, 0, :]
