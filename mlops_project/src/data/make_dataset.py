# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch import Tensor, randn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

# @click.command()
# click.argument('input_filepath', type=click.Path())
# @click.argument('output_filepath', type=click.Path())


def mnist():
    # exchange with the corrupted mnist dataset
    # Download and load the training data
    trainset = np.load("data/raw/corruptmnist/train_0.npz")
    images = Tensor(trainset["images"])  # transform to torch tensor
    labels = Tensor(trainset["labels"])
    my_dataset = TensorDataset(images, labels)
    train = DataLoader(my_dataset, batch_size=64, shuffle=True)
    testset = np.load("data/raw/corruptmnist/test.npz")
    images = Tensor(testset["images"])  # transform to torch tensor
    labels = Tensor(testset["labels"])
    my_dataset = TensorDataset(images, labels)
    test = DataLoader(my_dataset, batch_size=64, shuffle=True)
    torch.save(train, "data/processed/train_loader.pth")
    torch.save(test, "data/processed/test_loader.pth")


def main():
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    mnist()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
