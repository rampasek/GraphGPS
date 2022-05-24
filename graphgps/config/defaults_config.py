from torch_geometric.graphgym.register import register_config


def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    # Overwrite default dataset name
    cfg.dataset.name = 'none'
    
    # Overwrite default rounding precision
    cfg.round = 5


register_config('overwrite_defaults', overwrite_defaults_cfg)


def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # Directory path to a saved experiment, if set, take the model from there
    # and fine-tune it on a new dataset.
    cfg.train.finetune = ""
    # Freeze the pretrained part of the network, learning only the new head
    cfg.train.freeze_pretrained = False


register_config('extended_cfg', extended_cfg)
