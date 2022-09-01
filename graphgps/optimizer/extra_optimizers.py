import logging
import math
from typing import Iterator
from dataclasses import dataclass

import torch.optim as optim
from torch.nn import Parameter
from torch.optim import Adagrad, AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.graphgym.optim import SchedulerConfig
import torch_geometric.graphgym.register as register


@register.register_optimizer('adagrad')
def adagrad_optimizer(params: Iterator[Parameter], base_lr: float,
                      weight_decay: float) -> Adagrad:
    return Adagrad(params, lr=base_lr, weight_decay=weight_decay)


@register.register_optimizer('adamW')
def adamW_optimizer(params: Iterator[Parameter], base_lr: float,
                   weight_decay: float) -> AdamW:
    return AdamW(params, lr=base_lr, weight_decay=weight_decay)



@dataclass
class ExtendedSchedulerConfig(SchedulerConfig):
    reduce_factor: float = 0.5
    schedule_patience: int = 15
    min_lr: float = 1e-6
    num_warmup_epochs: int = 10
    train_mode: str = 'custom'
    eval_period: int = 1


@register.register_scheduler('plateau')
def plateau_scheduler(optimizer: Optimizer, patience: int,
                      lr_decay: float) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(optimizer, patience=patience, factor=lr_decay)


@register.register_scheduler('reduce_on_plateau')
def scheduler_reduce_on_plateau(optimizer: Optimizer, reduce_factor: float,
                                schedule_patience: int, min_lr: float,
                                train_mode: str, eval_period: int):
    if train_mode == 'standard':
        raise ValueError("ReduceLROnPlateau scheduler is not supported "
                         "by 'standard' graphgym training mode pipeline; "
                         "try setting config 'train.mode: custom'")

    if eval_period != 1:
        logging.warning("When config train.eval_period is not 1, the "
                        "optim.schedule_patience of ReduceLROnPlateau "
                        "may not behave as intended.")

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=reduce_factor,
        patience=schedule_patience,
        min_lr=min_lr,
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

    def modified_state_dict(ref):
        """Returns the state of the scheduler as a :class:`dict`.
        Additionally modified to ignore 'get_last_lr', 'state_dict'.
        Including these entries in the state dict would cause issues when
        loading a partially trained / pretrained model from a checkpoint.
        """
        return {key: value for key, value in ref.__dict__.items()
                if key not in ['sparsifier', 'get_last_lr', 'state_dict']}

    scheduler.state_dict = modified_state_dict.__get__(scheduler)

    return scheduler


@register.register_scheduler('linear_with_warmup')
def linear_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler


@register.register_scheduler('cosine_with_warmup')
def cosine_with_warmup_scheduler(optimizer: Optimizer,
                                 num_warmup_epochs: int, max_epoch: int):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_epochs,
        num_training_steps=max_epoch
    )
    return scheduler


def get_linear_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int,
        last_epoch: int = -1):
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
