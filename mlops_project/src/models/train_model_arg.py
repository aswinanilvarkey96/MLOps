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

import wandb
import click
import logging
import argparse
import wandb
log = logging.getLogger(__name__)

#@hydra.main(config_path = '../../src/config/',config_name="trainconfig")



def train():
    # Build your ArgumentParser however you like
    parser = argparse.ArgumentParser(description='Train Model')
    parser.add_argument('--learning_rate',type = float)
    parser.add_argument('--optimizer')

    # Get the hyperparameters
    args = parser.parse_args()

    trainpath= "../../data/processed/train_loader.pth"
    checkpoint  = "../..//models/checkpoint3.pth"
    epochs = 5
    lr = 3e-4

    # Pass them to wandb.init
    wandb.init(config=args)
    # Access all hyperparameter values through wandb.config
    config = wandb.config

    # Set up your model
    wandb.init(config=args)
    #trainpath = cfg.hyperparameters.trainpath
   # checkpoint = cfg.hyperparameters.checkpoint
    #epochs = cfg.hyperparameters.epochs
    #lr = cfg.hyperparameters.lr
    
    train_set = torch.load(trainpath)

    log.info("Training day and night")
    epochs = int(epochs)

    # Implement training loop here
    NN = MyAwesomeModel()
    criterion = nn.NLLLoss()
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(NN.parameters(), lr=args.learning_rate)
        
    else:
        optimizer = optim.Adam(NN.parameters(), lr=args.learning_rate)
    for e in range(epochs):
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
            log.info(f"Training loss: {running_loss/len(train_set)}")

    torch.save(NN.state_dict(),checkpoint)
    wandb.finish()


if __name__ == "__main__":
    

    train()
