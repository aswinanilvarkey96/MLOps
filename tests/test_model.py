
import os

import numpy as np
import torch
from torch import nn, optim
import sys
sys.path
sys.path.append(os.getcwd())
from src.models.model import MyAwesomeModel
def test_model():

    cwd = os.getcwd()

    train_set = torch.load(str(cwd) + "/data/processed/train_loader.pth")
    data_train, label_train =  next(iter(train_set))    
    NN = MyAwesomeModel()
    output = NN(data_train)
    assert output.shape == torch.Size([64,10]), 'Model output is incorrect'

if __name__ == "__main__":
    test_model()