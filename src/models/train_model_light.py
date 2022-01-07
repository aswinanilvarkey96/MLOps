import argparse
import logging
import sys

import click
import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from pytorch_lightning import LightningModule, Trainer
#from model import MyAwesomeModel
from torch import nn, optim

import wandb
from src.models.model_light import MyAwesomeModel

log = logging.getLogger(__name__)

@hydra.main(config_path = '../../src/config/',config_name="trainconfig")

def train(cfg):
    #wandb.init()
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
    """     for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            # Flatten MNIST images into a 784 long vector
            optimizer.zero_grad()
            output = NN(images)
            loss = criterion(output.squeeze(), labels.long())
            loss.backward()
            running_loss += loss.item()
            wandb.log({"train_loss": running_loss})
            optimizer.step()
        else:
            log.info(f"Training loss: {running_loss/len(train_set)}") """

    trainer = Trainer(max_epochs=5)
    trainer.fit(NN,train_set)
    torch.save(NN.state_dict(),str(get_original_cwd()) + checkpoint)
    #wandb.finish()


if __name__ == "__main__":
    train()
