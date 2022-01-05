import argparse
import sys

import numpy as np
import torch
#from model import MyAwesomeModel
from torch import nn, optim
import click

from src.models.model import MyAwesomeModel

import hydra
from hydra.utils import get_original_cwd

import logging
log = logging.getLogger(__name__)

@hydra.main(config_path = '../../src/config/',config_name="trainconfig")

def train(cfg):
    trainpath = cfg.hyperparameters.trainpath
    checkpoint = cfg.hyperparameters.checkpoint
    epochs = cfg.hyperparameters.epochs
    lr = cfg.hyperparameters.lr
    
    train_set = torch.load(str(get_original_cwd()) +trainpath)

    log.info("Training day and night")
    epochs = int(epochs)

    # Implement training loop here
    NN = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(NN.parameters(), lr=lr)
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            # Flatten MNIST images into a 784 long vector
            optimizer.zero_grad()
            output = NN(images)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
        else:
            log.info(f"Training loss: {running_loss/len(train_set)}")

    torch.save(NN.state_dict(),str(get_original_cwd()) + checkpoint)


if __name__ == "__main__":
    train()
