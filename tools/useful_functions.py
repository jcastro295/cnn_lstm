# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:56:49 2020

@author: Jhon
"""

import torch.nn as nn
import torch.optim as optim
def activation_func(activation):
    """
    List of activation functions

    Parameters:
    -----------
    activation: String
        Activation function to be selected

    Returns:
    --------
    activation function
    """
    return nn.ModuleDict([
        ['relu', nn.ReLU(inplace=True)],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01, inplace=True)],
        ['selu', nn.SELU(inplace=True)],
        ['none', nn.Identity()]
        ])[activation]


def set_learning_rate_scheduler(scheduler_name, optimizer, **kwargs):
    """
    Leaning rate options for training

    Parameters:
    -----------
    scheduler_name: String
        Name of the learning rate scheduler. Supported:
        "cosine"        - the cosine annealing scheduler
        "exponential"   - an exponential decay scheduler
        "step"          - a step scheduler (needs to be configured
                        through kwargs)
        "plateau"       - the plateau scheduler
        "none"          - disables scheduling by initializing a step
                        scheduler that never actually decreases the
                        learning rate
    optimizer: Obj
        Current optimizer

    Returns:
    --------
    scheduler: Obj
        Selected scheduler
    """

    scheduler_name = scheduler_name.lower()

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_name == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_name == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
    elif scheduler_name == "none":
        scheduler = optim.lr_scheduler.StepLR(optimizer, 100, gamma=1.0)
    else:
        raise KeyError("Sorry, but that scheduler isn't available")
    return scheduler


def set_optimizer(optimizer_name):
    """
    Optimizer for training

    Parameters:
    -----------
    optimizer_name: String
        Name of optimizer. Following options are supported.

    Returns:
    --------
    optimizer: Obj
        Selected optimizer
    """

    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == "adam":
        optimizer = optim.Adam
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW
    elif optimizer_name == "sgd":
        optimizer = optim.SGD
    elif optimizer_name == "adamax":
        optimizer = optim.Adamax
    else:
        raise KeyError("Sorry, but that optimizer isn't available")

    return optimizer

def set_criterion(criterion_name, **kwargs):
    """
    Set the criterium for training

    Parameters:
    -----------
    criterion_name: String
        Name of criterion. Supported:
        "crossentropy"  - cross entropy loss (classification)
        "mse"           - mean squared error loss (regression)
        "l1"            - L1 loss (regression)

    Return:
    -------
    criterion: Obj
        Selected criterion
    """
    c_name = criterion_name.lower()
    if c_name == "crossentropy":
        criterion = nn.CrossEntropyLoss(**kwargs)
    elif c_name == "mse":
        criterion = nn.MSELoss(**kwargs)
    elif c_name == 'l1':
        criterion = nn.L1Loss(**kwargs)
    else:
        raise KeyError(
            "Sorry, but that criterion isn't available. Please add it or check your spelling.")
    return criterion
