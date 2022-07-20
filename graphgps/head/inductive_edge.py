import numpy as np
import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head


@register_head('inductive_edge')
class GNNInductiveEdgeHead(nn.Module):
    """ GNN prediction head for inductive edge/link prediction tasks.

    Implementation adapted from the transductive GraphGym's GNNEdgeHead.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        # module to decode edges from node embeddings
        if cfg.model.edge_decoding == 'concat':
            self.layer_post_mp = MLP(
                new_layer_config(dim_in * 2, dim_out, cfg.gnn.layers_post_mp,
                                 has_act=False, has_bias=True, cfg=cfg))
            # requires parameter
            self.decode_module = lambda v1, v2: \
                self.layer_post_mp(torch.cat((v1, v2), dim=-1))
        else:
            if dim_out > 1:
                raise ValueError(
                    'Binary edge decoding ({})is used for multi-class '
                    'edge/link prediction.'.format(cfg.model.edge_decoding))
            self.layer_post_mp = MLP(
                new_layer_config(dim_in, dim_in, cfg.gnn.layers_post_mp,
                                 has_act=False, has_bias=True, cfg=cfg))
            if cfg.model.edge_decoding == 'dot':
                self.decode_module = lambda v1, v2: torch.sum(v1 * v2, dim=-1)
            elif cfg.model.edge_decoding == 'cosine_similarity':
                self.decode_module = nn.CosineSimilarity(dim=-1)
            else:
                raise ValueError(
                    f'Unknown edge decoding {cfg.model.edge_decoding}.')

    def _apply_index(self, batch):
        return batch.x[batch.edge_index_labeled], batch.edge_label

    def forward(self, batch):
        if cfg.model.edge_decoding != 'concat':
            batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        nodes_first = pred[0]
        nodes_second = pred[1]
        pred = self.decode_module(nodes_first, nodes_second)
        if not self.training:  # Compute extra stats when in evaluation mode.
            stats = self.compute_mrr(batch)
            return pred, label, stats
        else:
            return pred, label

    def compute_mrr(self, batch):
        if cfg.model.edge_decoding != 'dot':
            raise ValueError(
                f'Unsupported edge decoding {cfg.model.edge_decoding}.')

        stats = {}
        for data in batch.to_data_list():
            # print(data.num_nodes)
            # print(data.edge_index_labeled)
            # print(data.edge_label)
            pred = data.x @ data.x.transpose(0, 1)
            # print(pred.shape)

            pos_edge_index = data.edge_index_labeled[:, data.edge_label == 1]
            num_pos_edges = pos_edge_index.shape[1]
            # print(pos_edge_index, num_pos_edges)

            pred_pos = pred[pos_edge_index[0], pos_edge_index[1]]
            # print(pred_pos)

            if num_pos_edges > 0:
                neg_mask = torch.ones([num_pos_edges, data.num_nodes],
                                      dtype=torch.bool)
                neg_mask[torch.arange(num_pos_edges), pos_edge_index[1]] = False
                pred_neg = pred[pos_edge_index[0]][neg_mask].view(num_pos_edges, -1)
                # print(pred_neg, pred_neg.shape)
                mrr_list = self._eval_mrr(pred_pos, pred_neg, 'torch')
            else:
                # Return empty stats.
                mrr_list = self._eval_mrr(pred_pos, pred_pos, 'torch')

            # print(mrr_list)
            for key, val in mrr_list.items():
                if key.endswith('_list'):
                    key = key[:-len('_list')]
                    val = float(val.mean().item())
                if np.isnan(val):
                    val = 0.
                if key not in stats:
                    stats[key] = [val]
                else:
                    stats[key].append(val)
                # print(key, val)
            # print('-' * 80)

        # print('=' * 80, batch.split)
        batch_stats = {}
        for key, val in stats.items():
            mean_val = sum(val) / len(val)
            batch_stats[key] = mean_val
            # print(f"{key}: {mean_val}")
        return batch_stats

    def _eval_mrr(self, y_pred_pos, y_pred_neg, type_info):
        """ Compute Hits@k and Mean Reciprocal Rank (MRR).

        Implementation from OGB:
        https://github.com/snap-stanford/ogb/blob/master/ogb/linkproppred/evaluate.py

        Args:
            y_pred_neg: array with shape (batch size, num_entities_neg).
            y_pred_pos: array with shape (batch size, )
        """

        if type_info == 'torch':
            y_pred = torch.cat([y_pred_pos.view(-1, 1), y_pred_neg], dim=1)
            argsort = torch.argsort(y_pred, dim=1, descending=True)
            ranking_list = torch.nonzero(argsort == 0, as_tuple=False)
            ranking_list = ranking_list[:, 1] + 1
            hits1_list = (ranking_list <= 1).to(torch.float)
            hits3_list = (ranking_list <= 3).to(torch.float)
            hits10_list = (ranking_list <= 10).to(torch.float)
            mrr_list = 1. / ranking_list.to(torch.float)

            return {'hits@1_list': hits1_list,
                    'hits@3_list': hits3_list,
                    'hits@10_list': hits10_list,
                    'mrr_list': mrr_list}

        else:
            y_pred = np.concatenate([y_pred_pos.reshape(-1, 1), y_pred_neg],
                                    axis=1)
            argsort = np.argsort(-y_pred, axis=1)
            ranking_list = (argsort == 0).nonzero()
            ranking_list = ranking_list[1] + 1
            hits1_list = (ranking_list <= 1).astype(np.float32)
            hits3_list = (ranking_list <= 3).astype(np.float32)
            hits10_list = (ranking_list <= 10).astype(np.float32)
            mrr_list = 1. / ranking_list.astype(np.float32)

            return {'hits@1_list': hits1_list,
                    'hits@3_list': hits3_list,
                    'hits@10_list': hits10_list,
                    'mrr_list': mrr_list}
