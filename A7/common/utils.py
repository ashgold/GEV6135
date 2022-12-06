"""
General utilities to help with implementation
"""
import random
import torch


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def count_num_layers(model, blocks=['block', 'fc', 'net'], shortcut='shortcut',
                     layers=['Conv2d', 'Linear']):
    """
    Count the number of specified layers in the given model.
    Layers in shortcut are not counted.

    Inputs:
    - model: PyTorch module to count the number of layers
    - block: Name of block
    - shortcut: Name of shortcut
    - layers: List of the name of layers to be counted

    Returns:
    - count: number of layers counted
    """
    model_str = str(model)
    count = 0
    skip = False
    for line in model_str.splitlines():
        if shortcut in line:
            skip = True
        else:
            for block in blocks:
                if block in line:
                    skip = False
                    break
        if not skip:
            for layer in layers:
                if layer in line:
                    count += 1
    return count


def count_num_params(model):
    """
    Count the number of learnable parameters in the given model.

    Inputs:
    - model: PyTorch module to count the number of learnable parameters

    Returns: Number of learnable parameters counted
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
