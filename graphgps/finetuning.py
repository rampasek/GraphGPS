import logging
import os
import os.path as osp

import torch
from torch_geometric.graphgym.config import set_cfg
from yacs.config import CfgNode


def get_final_pretrained_ckpt(ckpt_dir):
    if osp.exists(ckpt_dir):
        names = os.listdir(ckpt_dir)
        epochs = [int(name.split('.')[0]) for name in names]
        final_epoch = max(epochs)
    else:
        raise FileNotFoundError(f"Pretrained model dir not found: {ckpt_dir}")
    return osp.join(ckpt_dir, f'{final_epoch}.ckpt')


def compare_cfg(cfg_main, cfg_secondary, field_name, strict=False):
    main_val, secondary_val = cfg_main, cfg_secondary
    for f in field_name.split('.'):
        main_val = main_val[f]
        secondary_val = secondary_val[f]
    if main_val != secondary_val:
        if strict:
            raise ValueError(f"Main and pretrained configs must match on "
                             f"'{field_name}'")
        else:
            logging.warning(f"Pretrained models '{field_name}' differs, "
                            f"using: {main_val}")


def set_new_cfg_allowed(config, is_new_allowed):
    """ Set YACS config (and recursively its subconfigs) to allow merging
        new keys from other configs.
    """
    config.__dict__[CfgNode.NEW_ALLOWED] = is_new_allowed
    # Recursively set new_allowed state
    for v in config.__dict__.values():
        if isinstance(v, CfgNode):
            set_new_cfg_allowed(v, is_new_allowed)
    for v in config.values():
        if isinstance(v, CfgNode):
            set_new_cfg_allowed(v, is_new_allowed)


def load_pretrained_model_cfg(cfg):
    pretrained_cfg_fname = osp.join(cfg.pretrained.dir, 'config.yaml')
    if not os.path.isfile(pretrained_cfg_fname):
        FileNotFoundError(f"Pretrained model config not found: "
                          f"{pretrained_cfg_fname}")

    logging.info(f"[*] Updating cfg from pretrained model: "
                 f"{pretrained_cfg_fname}")

    pretrained_cfg = CfgNode()
    set_cfg(pretrained_cfg)
    set_new_cfg_allowed(pretrained_cfg, True)
    pretrained_cfg.merge_from_file(pretrained_cfg_fname)

    assert cfg.model.type == 'GPSModel', \
        "Fine-tuning regime is untested for other model types."
    compare_cfg(cfg, pretrained_cfg, 'model.type', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'model.graph_pooling')
    compare_cfg(cfg, pretrained_cfg, 'model.edge_decoding')
    compare_cfg(cfg, pretrained_cfg, 'dataset.node_encoder', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.node_encoder_name', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.node_encoder_bn', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.edge_encoder', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.edge_encoder_name', strict=True)
    compare_cfg(cfg, pretrained_cfg, 'dataset.edge_encoder_bn', strict=True)

    # Copy over all PE/SE configs
    for key in cfg.keys():
        if key.startswith('posenc_'):
            cfg[key] = pretrained_cfg[key]

    # Copy over GT config
    cfg.gt = pretrained_cfg.gt

    # Copy over GNN cfg but not those for the prediction head
    compare_cfg(cfg, pretrained_cfg, 'gnn.head')
    compare_cfg(cfg, pretrained_cfg, 'gnn.layers_post_mp')
    compare_cfg(cfg, pretrained_cfg, 'gnn.act')
    compare_cfg(cfg, pretrained_cfg, 'gnn.dropout')
    head = cfg.gnn.head
    post_mp = cfg.gnn.layers_post_mp
    act = cfg.gnn.act
    drp = cfg.gnn.dropout
    cfg.gnn = pretrained_cfg.gnn
    cfg.gnn.head = head
    cfg.gnn.layers_post_mp = post_mp
    cfg.gnn.act = act
    cfg.gnn.dropout = drp
    return cfg


def init_model_from_pretrained(model, pretrained_dir,
                               freeze_main=False, reset_prediction_head=True):
    """ Copy model parameters from pretrained model except the prediction head.

    Args:
        model: Initialized model with random weights.
        pretrained_dir: Root directory of saved pretrained model.
        freeze_main: If True, do not finetune the loaded pretrained parameters
            of the `main body` (train the prediction head only), else train all.
        reset_prediction_head: If True, reset parameters of the prediction head,
            else keep the pretrained weights.

    Returns:
        Updated pytorch model object.
    """
    from torch_geometric.graphgym.checkpoint import MODEL_STATE

    ckpt_file = get_final_pretrained_ckpt(osp.join(pretrained_dir, '0', 'ckpt'))
    logging.info(f"[*] Loading from pretrained model: {ckpt_file}")

    ckpt = torch.load(ckpt_file)
    pretrained_dict = ckpt[MODEL_STATE]
    model_dict = model.state_dict()

    # print('>>>> pretrained dict: ')
    # print(pretrained_dict.keys())
    # print('>>>> model dict: ')
    # print(model_dict.keys())

    if reset_prediction_head:
        # Filter out prediction head parameter keys.
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if not k.startswith('post_mp')}
    # Overwrite entries in the existing state dict.
    model_dict.update(pretrained_dict)
    # Load the new state dict.
    model.load_state_dict(model_dict)

    if freeze_main:
        for key, param in model.named_parameters():
            if not key.startswith('post_mp'):
                param.requires_grad = False
    return model
