import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('subtoken_cross_entropy')
def subtoken_cross_entropy(pred_list, true):
    """Subtoken prediction cross-entropy loss for ogbg-code2.
    """
    if cfg.dataset.task_type == 'subtoken_prediction':
        if cfg.model.loss_fun != 'cross_entropy':
            raise ValueError("Only 'cross_entropy' loss_fun supported with "
                             "'subtoken_prediction' task_type.")
        multicls_criterion = torch.nn.CrossEntropyLoss()
        loss = 0
        for i in range(len(pred_list)):
            loss += multicls_criterion(pred_list[i].to(torch.float32), true['y_arr'][:, i])
        loss = loss / len(pred_list)

        return loss, pred_list
