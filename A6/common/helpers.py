"""
Helper functions used in Assignment 5
"""
import torch
import matplotlib.pyplot as plt
import math


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print("Hello from helpers.py!")


def plot_stats(stat_dict):
    # Plot the loss function and train / validation accuracies
    plt.subplot(1, 2, 1)
    plt.plot(stat_dict['loss_history'], 'o')
    plt.title('Loss history')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(stat_dict['train_acc_history'], 'o-', label='train')
    plt.plot(stat_dict['val_acc_history'], 'o-', label='val')
    plt.title('Classification accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(14, 4)
    plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    # print(Xs.shape)
    grid_size = int(math.ceil(math.sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = torch.zeros((grid_height, grid_width, C), device=Xs.device)
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = torch.min(img), torch.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    return grid


# Visualize the weights of the network
def show_net_weights(net):
    W1 = net.params['W1']
    W1 = W1.reshape(3, 32, 32, -1).transpose(0, 3)
    plt.imshow(visualize_grid(W1, padding=3).type(torch.uint8).cpu())
    plt.gca().axis('off')
    plt.show()


def plot_acc_curves(stat_dict):
    plt.subplot(1, 2, 1)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats['train_acc_history'], label=str(key))
    plt.title('Train accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')

    plt.subplot(1, 2, 2)
    for key, single_stats in stat_dict.items():
        plt.plot(single_stats['val_acc_history'], label=str(key))
    plt.title('Validation accuracy history')
    plt.xlabel('Epoch')
    plt.ylabel('Classification accuracy')
    plt.legend()

    plt.gcf().set_size_inches(14, 5)
    plt.show()


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: A tensor of shape (N, C) containing a minibatch of data,
         where x[i, j] is the score for the j-th class for the i-th input.
    - y: A tensor of shape (N,) containing labels; y[i] is the label
         for x[i] and 0 <= y[i] < C

    Returns a tuple of:
    - loss: a scalar float
    - dx: Gradient of the loss with respect to x
    """
    tx = x.detach()
    tx.requires_grad = True
    loss = torch.nn.functional.cross_entropy(tx, y.to(device=tx.device))
    try:
        loss.backward()
        dx = tx.grad.detach()
    except RuntimeError:
        dx = torch.zeros_like(tx)
    return loss.detach(), dx
