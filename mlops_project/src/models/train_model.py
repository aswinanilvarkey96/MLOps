import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
#from model import MyAwesomeModel
from torch import nn, optim
import click

from src.models.model import MyAwesomeModel

@click.command()
@click.argument('epochs', default=20)
@click.argument('data', default='train_loader')

def train(epochs, data):
    train_set = torch.load("data/processed/" + data + ".pth")

    print("Training day and night")
    epochs = int(epochs)

    # Implement training loop here
    NN = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(NN.parameters(), lr=0.003)
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            # Flatten MNIST images into a 784 long vector
            optimizer.zero_grad()
            # TODO: Training pass
            output = NN(images)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        else:
            print(f"Training loss: {running_loss/len(train_set)}")

    torch.save(NN.state_dict(), "models/checkpoint.pth")


if __name__ == "__main__":
    train()
