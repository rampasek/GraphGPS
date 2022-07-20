"""
SignNet https://arxiv.org/abs/2202.13013
based on https://github.com/cptq/SignNet-BasisNet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torch_geometric.nn import GINConv
from torch_scatter import scatter


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 use_bn=False, use_ln=False, dropout=0.5, activation='relu',
                 residual=False):
        super().__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()

        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation')
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
        self.residual = residual

    def forward(self, x):
        x_prev = x
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2, 1)).transpose(2, 1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            if self.residual and x_prev.shape == x.shape: x = x + x_prev
            x = F.dropout(x, p=self.dropout, training=self.training)
            x_prev = x
        x = self.lins[-1](x)
        if self.residual and x_prev.shape == x.shape:
            x = x + x_prev
        return x


class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers,
                 use_bn=True, dropout=0.5, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels,
                             2, use_bn=use_bn, dropout=dropout,
                             activation=activation)
            self.layers.append(GINConv(update_net))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2,
                         use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i - 1](x)
                    elif x.ndim == 3:
                        x = self.bns[i - 1](x.transpose(2, 1)).transpose(2, 1)
                    else:
                        raise ValueError('invalid x dim')
            x = layer(x, edge_index)
        return x


class GINDeepSigns(nn.Module):
    """ Sign invariant neural network with MLP aggregation.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 k, dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        rho_dim = out_channels * k
        self.rho = MLP(rho_dim, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        x = x.transpose(0, 1) # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)
        x = x.transpose(0, 1).reshape(N, -1)  # K x N x Out -> N x (K * Out)
        x = self.rho(x)  # N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


class MaskedGINDeepSigns(nn.Module):
    """ Sign invariant neural network with sum pooling and DeepSet.
        f(v1, ..., vk) = rho(enc(v1) + enc(-v1), ..., enc(vk) + enc(-vk))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dim_pe, rho_num_layers, use_bn=False, use_ln=False,
                 dropout=0.5, activation='relu'):
        super().__init__()
        self.enc = GIN(in_channels, hidden_channels, out_channels, num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)
        self.rho = MLP(out_channels, hidden_channels, dim_pe, rho_num_layers,
                       use_bn=use_bn, dropout=dropout, activation=activation)

    def batched_n_nodes(self, batch_index):
        batch_size = batch_index.max().item() + 1
        one = batch_index.new_ones(batch_index.size(0))
        n_nodes = scatter(one, batch_index, dim=0, dim_size=batch_size,
                          reduce='add')  # Number of nodes in each graph.
        n_nodes = n_nodes.unsqueeze(1)
        return torch.cat([size * n_nodes.new_ones(size) for size in n_nodes])

    def forward(self, x, edge_index, batch_index):
        N = x.shape[0]  # Total number of nodes in the batch.
        K = x.shape[1]  # Max. number of eigen vectors / frequencies.
        x = x.transpose(0, 1)  # N x K x In -> K x N x In
        x = self.enc(x, edge_index) + self.enc(-x, edge_index)  # K x N x Out
        x = x.transpose(0, 1)  # K x N x Out -> N x K x Out

        batched_num_nodes = self.batched_n_nodes(batch_index)
        mask = torch.cat([torch.arange(K).unsqueeze(0) for _ in range(N)])
        mask = (mask.to(batch_index.device) < batched_num_nodes.unsqueeze(1)).bool()
        # print(f"     - mask: {mask.shape} {mask}")
        # print(f"     - num_nodes: {num_nodes}")
        # print(f"     - batched_num_nodes: {batched_num_nodes.shape} {batched_num_nodes}")
        x[~mask] = 0
        x = x.sum(dim=1)  # (sum over K) -> N x Out
        x = self.rho(x)  # N x Out -> N x dim_pe (Note: in the original codebase dim_pe is always K)
        return x


@register_node_encoder('SignNet')
class SignNetNodeEncoder(torch.nn.Module):
    """SignNet Positional Embedding node encoder.
    https://arxiv.org/abs/2202.13013
    https://github.com/cptq/SignNet-BasisNet

    Uses precomputated Laplacian eigen-decomposition, but instead
    of eigen-vector sign flipping + DeepSet/Transformer, computes the PE as:
    SignNetPE(v_1, ... , v_k) = \rho ( [\phi(v_i) + \rhi(−v_i)]^k_i=1 )
    where \phi is GIN network applied to k first non-trivial eigenvectors, and
    \rho is an MLP if k is a constant, but if all eigenvectors are used then
    \rho is DeepSet with sum-pooling.

    SignNetPE of size dim_pe will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with SignNetPE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()
        dim_in = cfg.share.dim_in  # Expected original input node features dim

        pecfg = cfg.posenc_SignNet
        dim_pe = pecfg.dim_pe  # Size of PE embedding
        model_type = pecfg.model  # Encoder NN model type for SignNet
        if model_type not in ['MLP', 'DeepSet']:
            raise ValueError(f"Unexpected SignNet model {model_type}")
        self.model_type = model_type
        sign_inv_layers = pecfg.layers  # Num. layers in \phi GNN part
        rho_layers = pecfg.post_layers  # Num. layers in \rho MLP/DeepSet
        if rho_layers < 1:
            raise ValueError(f"Num layers in rho model has to be positive.")
        max_freqs = pecfg.eigen.max_freqs  # Num. eigenvectors (frequencies)
        self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 1:
            raise ValueError(f"SignNet PE size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        # Sign invariant neural network.
        if self.model_type == 'MLP':
            self.sign_inv_net = GINDeepSigns(
                in_channels=1,
                hidden_channels=pecfg.phi_hidden_dim,
                out_channels=pecfg.phi_out_dim,
                num_layers=sign_inv_layers,
                k=max_freqs,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        elif self.model_type == 'DeepSet':
            self.sign_inv_net = MaskedGINDeepSigns(
                in_channels=1,
                hidden_channels=pecfg.phi_hidden_dim,
                out_channels=pecfg.phi_out_dim,
                num_layers=sign_inv_layers,
                dim_pe=dim_pe,
                rho_num_layers=rho_layers,
                use_bn=True,
                dropout=0.0,
                activation='relu'
            )
        else:
            raise ValueError(f"Unexpected model {self.model_type}")

    def forward(self, batch):
        if not (hasattr(batch, 'eigvals_sn') and hasattr(batch, 'eigvecs_sn')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_SignNet.enable' to True")
        # eigvals = batch.eigvals_sn
        eigvecs = batch.eigvecs_sn

        # pos_enc = torch.cat((eigvecs.unsqueeze(2), eigvals), dim=2)  # (Num nodes) x (Num Eigenvectors) x 2
        pos_enc = eigvecs.unsqueeze(-1)  # (Num nodes) x (Num Eigenvectors) x 1

        empty_mask = torch.isnan(pos_enc)
        pos_enc[empty_mask] = 0  # (Num nodes) x (Num Eigenvectors) x 1

        # SignNet
        pos_enc = self.sign_inv_net(pos_enc, batch.edge_index, batch.batch)  # (Num nodes) x (pos_enc_dim)

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            batch.pe_SignNet = pos_enc
        return batch
