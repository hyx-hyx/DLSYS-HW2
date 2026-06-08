import os
import sys
import time

import needle as ndl
import needle.nn as nn
import numpy as np

from python.needle import data
from python.needle.data.data_basic import DataLoader
from python.needle.data.datasets import mnist_dataset
from python.needle.data.datasets.mnist_dataset import MNISTDataset
from tests.hw2.test_nn_and_optim import softmax_loss_forward

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
        model_list.append(ResidualBlock(
            hidden_dim, hidden_dim // 2, norm, drop_prob))
    model_list.append(nn.Linear(hidden_dim, num_classes))

    return nn.Sequential(*(tuple(model_list)))
    # END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    loss = 0
    error = 0
    for i, batch in enumerate(dataloader):
        batch_x, batch_y = batch[0], batch[1]

        num_examples, input_dim = batch_x.shape
        num_classes = batch_y.shape[0]
        logits = model.forward(batch_x)
        error += (logits != batch_y)
        f = ndl.nn.SoftmaxLoss()
        loss += f(logits, batch_y)
        loss.backward()
        opt.step()
    return error, loss
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
    data_mnist_dataset = MNISTDataset(
        "train-images-idx3-ubyte", "train-labels-idx1-ubyte")
    test_mnist_dataset = MNISTDataset(
        "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte.gz")
    dataloader = DataLoader(data_mnist_dataset, batch_size=batch_size)
    testloader = DataLoader(test_mnist_dataset, batch_size=batch_size)
    model = MLPResNet(784, hidden_dim)

    for _ in range(epochs):
        epoch(dataloader, model=model, opt=optimizer)
    # END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
