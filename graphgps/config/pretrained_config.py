from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('cfg_pretrained')
def set_cfg_pretrained(cfg):
    """Configuration options for loading a pretrained model.
    """

    cfg.pretrained = CN()

    # Directory path to a saved experiment, if set, load the model from there
    # and fine-tune / run inference with it on a specified dataset.
    cfg.pretrained.dir = ""

    # Discard pretrained weights of the prediction head and reinitialize.
    cfg.pretrained.reset_prediction_head = True

    # Freeze the main pretrained 'body' of the model, learning only the new head
    cfg.pretrained.freeze_main = False
