import os
from collections import deque
import numpy as np
import torch
import torch.nn as nn


ndarray = np.ndarray
Tensor = torch.Tensor


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_model(model, path, name):
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path+name+".pt")


def load_model(model, path, name, device):
    model.load_state_dict(torch.load(path+name+".pt", map_location=device), strict=False)


class BatchMovingAverage():
    """Computes moving average over the last `k` mini-batches
    and stores the smallest recorded moving average in `min_avg`."""
    def __init__(self, k: int) -> None:
        self.values = deque([], maxlen=k)
        self.min_avg = np.inf

    def add_value(self, value: float) -> None:
        self.values.append(value)

    def get_average(self) -> float:
        if len(self.values) == 0:
            avg = np.nan
        else:
            avg = sum(self.values) / len(self.values)

        if avg < self.min_avg:
            self.min_avg = avg

        return avg

    def get_min_average(self):
        return self.min_avg


def kl_norm_norm(mu0: Tensor, mu1: Tensor, sig0: Tensor, sig1: Tensor) -> Tensor:
    """Calculates KL divergence between two K-dimensional Normal
        distributions with diagonal covariance matrices.

    Args:
        mu0: Mean of the first distribution. Has shape (*, K).
        mu1: Mean of the second distribution. Has shape (*, K).
        sig0: Diagonal of the covatiance matrix of the first distribution. Has shape (*, K).
        sig1: Diagonal of the covatiance matrix of the second distribution. Has shape (*, K).

    Returns:
        KL divergence between the distributions. Has shape (*, 1).
    """
    assert mu0.shape == mu1.shape == sig0.shape == sig1.shape, (f"{mu0.shape=} {mu1.shape=} {sig0.shape=} {sig1.shape=}")
    a = (sig0 / sig1).pow(2).sum(-1, keepdim=True)
    b = ((mu1 - mu0).pow(2) / sig1**2).sum(-1, keepdim=True)
    c = 2 * (torch.log(sig1) - torch.log(sig0)).sum(-1, keepdim=True)
    kl = 0.5 * (a + b + c - mu0.shape[-1])
    return kl


def init_weights(m):
    if isinstance(m, nn.Linear):
        pass
        # torch.nn.init.xavier_uniform(m.weight)
        # if m.bias is not None:
        #     m.bias.data.fill_(0.0)


# def create_mlp(input_size, output_size, hid_size, hid_layers, nonlin):
#     layers = [nn.Linear(input_size, hid_size), nonlin()]
#     for _ in range(hid_layers - 1):
#         layers.append(nn.Linear(hid_size, hid_size))
#         layers.append(nonlin())
#     layers.append(nn.Linear(hid_size, output_size))
#     return nn.Sequential(*layers)


def create_mlp(
        input_size,
        output_size,
        hidden_size,
        num_hidden_layers,
        activation_func,
        use_layer_norm=False,
        use_dropout=False,
        dropout_prob=0.5,
):
    """
    Create MLP with optional layer normalization and dropout.

    Args:
        input_size (int): The size of the input layer.
        output_size (int): The size of the output layer.
        hidden_size (int): The size of the hidden layers.
        num_hidden_layers (int): The number of hidden layers.
        activation_func (function): The nonlinear activation function to use.
        use_layer_norm (bool): Whether to use layer normalization (default: False).
        use_dropout (bool): Whether to use dropout (default: False).
        dropout_prob (float): Dropout probability, used if use_dropout is True (default: 0.5).

    Returns:
        nn.Sequential: The constructed MLP model.
    """
    layers = []
    for i in range(num_hidden_layers):
        if i == 0:
            layers.append(nn.Linear(input_size, hidden_size))
        else:
            layers.append(nn.Linear(hidden_size, hidden_size))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(activation_func())
        if use_dropout:
            layers.append(nn.Dropout(dropout_prob))

    layers.append(nn.Linear(hidden_size, output_size))
    return nn.Sequential(*layers)


def pad_context(context: list[ndarray]) -> tuple[ndarray, ndarray]:
    """
    Pads a list of arrays to the same length and creates a mask.

    Args:
        context (list of np.array): A list of 2D numpy arrays of varying lengths.

    Returns:
        tuple: A tuple containing:
            - padded_context (np.array): A 3D numpy array where each 2D array is padded to the same length.
            - mask (np.array): A 2D mask array with -inf values for padding and an extra column for an aggregation token.
    """
    # ctx_size = 0.75  # 1=full ctx, 0.5=half ctx
    # new_context: list[ndarray] = []
    # for i in range(len(context)):
    #     mask = (context[i][:, 0] + t_ctx) >= (t_ctx * (1 - ctx_size))
    #     new_context.append(context[i][mask])
    # context = new_context

    max_len = max(len(c) for c in context)
    mask = np.zeros((len(context), max_len + 1), dtype=np.float32)

    for i, c in enumerate(context):
        mask[i, len(c):-1] = -np.inf
        context[i] = np.pad(c, pad_width=((0, max_len - len(c)), (0, 0)), constant_values=-1)  # type: ignore

    padded_context = np.stack(context)
    return padded_context, mask


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
