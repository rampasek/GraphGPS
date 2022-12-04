from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_sem')
def set_cfg_sem(cfg):
    """Configuration for Graph Transformer-style models, e.g.:
    - Spectral Attention Network (SAN) Graph Transformer.
    - "vanilla" Transformer / Performer.
    - General Powerful Scalable (GPS) Model.
    """

    # Positional encodings argument group
    cfg.sem = CN()

    # L number of simplicial embeddings
    cfg.sem.L = 64

    # V dimension of each simplicial embeddings
    cfg.sem.V = 8

    # SEM temperature
    cfg.sem.tau = 1.
