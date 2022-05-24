import math

import torch.optim as optim
from torch.optim import Optimizer
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.optimizer import OptimizerConfig, SchedulerConfig
from torch_geometric.graphgym.register import (register_optimizer,
                                               register_scheduler)


def optimizer_adagrad(params, optimizer_config: OptimizerConfig):
    if optimizer_config.optimizer == 'adagrad':
        optimizer = optim.Adagrad(params, lr=optimizer_config.base_lr,
                                  weight_decay=optimizer_config.weight_decay)
        return optimizer


register_optimizer('adagrad', optimizer_adagrad)


def optimizer_adamW(params, optimizer_config: OptimizerConfig):
    if optimizer_config.optimizer == 'adamW':
        optimizer = optim.AdamW(params, lr=optimizer_config.base_lr,
                                weight_decay=optimizer_config.weight_decay)
        return optimizer


register_optimizer('adamW', optimizer_adamW)


def scheduler_reduce_on_plateau(optimizer, scheduler_config: SchedulerConfig):
    if scheduler_config.scheduler == 'reduce_on_plateau':
        if cfg.train.mode == 'standard':
            raise ValueError("ReduceLROnPlateau scheduler is not supported "
                             "by 'standard' graphgym training mode pipeline; "
                             "try setting config 'train.mode: custom'")
        if cfg.train.eval_period != 1:
            raise ValueError("When config train.eval_period is not 1, the "
                             "optim.schedule_patience of ReduceLROnPlateau "
                             "doesn't behave as intended.")
        metric_mode = 'min'
        # metric_mode = cfg.metric_agg[-3:]
        # if metric_mode not in ['min', 'max']:
        #     raise ValueError(f"Failed to automatically infer min or max mode "
        #                      f"from cfg.metric_agg='{cfg.metric_agg}'")

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=metric_mode,
            factor=cfg.optim.reduce_factor,
            patience=cfg.optim.schedule_patience,
            min_lr=cfg.optim.min_lr,
            verbose=True
        )
        if not hasattr(scheduler, 'get_last_lr'):
            # ReduceLROnPlateau doesn't have `get_last_lr` method as of current
            # pytorch1.10; we add it here for consistency with other schedulers.
            def get_last_lr(self):
                """ Return last computed learning rate by current scheduler.
                """
                return self._last_lr

            scheduler.get_last_lr = get_last_lr.__get__(scheduler)
            scheduler._last_lr = [group['lr']
                                  for group in scheduler.optimizer.param_groups]

        return scheduler


register_scheduler('reduce_on_plateau', scheduler_reduce_on_plateau)


def scheduler_with_warmup(optimizer, scheduler_config: SchedulerConfig):
    scheduler = None
    if scheduler_config.scheduler == 'linear_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.optim.num_warmup_epochs,
            num_training_steps=scheduler_config.max_epoch
        )
    elif scheduler_config.scheduler == 'cosine_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=cfg.optim.num_warmup_epochs,
            num_training_steps=scheduler_config.max_epoch
        )
    if scheduler is not None:
        return scheduler

register_scheduler('scheduler_with_warmup', scheduler_with_warmup)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                    num_training_steps, last_epoch=-1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases linearly from the
    initial lr set in the optimizer to 0, after a warmup period during which it
    increases linearly from 0 to the initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        num_cycles: float = 0.5, last_epoch: int = -1):
    """
    Implementation by Huggingface:
    https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/optimization.py

    Create a schedule with a learning rate that decreases following the values
    of the cosine function between the initial lr set in the optimizer to 0,
    after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just
            decrease from the max value to 0 following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(1e-6, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
