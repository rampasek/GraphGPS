import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder('EquivStableLapPE')
class EquivStableLapPENodeEncoder(torch.nn.Module):
    """Equivariant and Stable Laplace Positional Embedding node encoder.

    This encoder simply transforms the k-dim node LapPE to d-dim to be
    later used at the local GNN module as edge weights.
    Based on the approach proposed in paper https://openreview.net/pdf?id=e95i1IHcWj
    
    Args:
        dim_emb: Size of final node embedding
    """

    def __init__(self, dim_emb):
        super().__init__()

        pecfg = cfg.posenc_EquivStableLapPE
        max_freqs = pecfg.eigen.max_freqs  # Num. eigenvectors (frequencies)
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(max_freqs)
        else:
            self.raw_norm = None

        self.linear_encoder_eigenvec = nn.Linear(max_freqs, dim_emb)

    def forward(self, batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; set "
                             f"config 'posenc_EquivStableLapPE.enable' to True")
        pos_enc = batch.EigVecs

        empty_mask = torch.isnan(pos_enc)  # (Num nodes) x (Num Eigenvectors)
        pos_enc[empty_mask] = 0.  # (Num nodes) x (Num Eigenvectors)

        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)

        pos_enc = self.linear_encoder_eigenvec(pos_enc)

        # Keep PE separate in a variable
        batch.pe_EquivStableLapPE = pos_enc

        return batch
