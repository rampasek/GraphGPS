from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_graphormer')
def set_cfg_gt(cfg):
    cfg.graphormer = CN()
    cfg.graphormer.num_layers = 6
    cfg.graphormer.embed_dim = 80
    cfg.graphormer.num_heads = 4
    cfg.graphormer.dropout = 0.0
    cfg.graphormer.attention_dropout = 0.0
    cfg.graphormer.mlp_dropout = 0.0
    cfg.graphormer.input_dropout = 0.0
    cfg.graphormer.use_graph_token = True

    cfg.posenc_GraphormerBias = CN()
    cfg.posenc_GraphormerBias.enable = False
    cfg.posenc_GraphormerBias.node_degrees_only = False
    cfg.posenc_GraphormerBias.dim_pe = 0
    cfg.posenc_GraphormerBias.num_spatial_types = None
    cfg.posenc_GraphormerBias.num_in_degrees = None
    cfg.posenc_GraphormerBias.num_out_degrees = None
