
import torch
from torch import nn, optim
import numpy as np
from src.models.model import MyAwesomeModel

train_set = torch.load("data/processed/train_loader.pth")
data_train, label_train =  next(iter(train_set))    
NN = MyAwesomeModel()
output = NN(data_train)
assert output.shape == torch.Size([64,10]), 'Model output is incorrect'