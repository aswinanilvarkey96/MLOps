
import torch
from torch import nn, optim
import numpy as np
import os
cwd = os.getcwd()
os.chdir(str(cwd) + "/src/models/")
ch = str(cwd) + "/src/models/"
print(os.getcwd())
from home.runner.worl.MLOps.MLOps.src.models.model import MyAwesomeModel
os.chdir(str(cwd))
train_set = torch.load(str(cwd) + "/data/processed/train_loader.pth")
data_train, label_train =  next(iter(train_set))    
NN = MyAwesomeModel()
output = NN(data_train)
assert output.shape == torch.Size([64,10]), 'Model output is incorrect'