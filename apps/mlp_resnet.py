import os
import sys
import time

import needle as ndl
import needle.nn as nn
import numpy as np

sys.path.append("../python")


np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    # BEGIN YOUR SOLUTION
    seq = nn.Sequential(
        nn.Linear(
            dim,
            hidden_dim),
        norm(hidden_dim),
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(
            hidden_dim,
            dim),
        norm(dim)
    )
    return nn.Sequential(nn.Residual(seq), nn.ReLU())
    # END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    # BEGIN YOUR SOLUTION
    model_list = []
    model_list.append(nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU()))
    for _ in range(num_blocks):
        model_list.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    model_list.append(nn.Linear(hidden_dim, num_classes))

    return nn.Sequential(*(tuple(model_list)))
    # END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    raise NotImplementedError()
    # END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    raise NotImplementedError()
    # END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
