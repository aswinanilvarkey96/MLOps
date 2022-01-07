
import torch
from torch import nn, optim
import numpy as np

train_set = torch.load("data/processed/train_loader.pth")
assert len(train_set) == 79, 'Training set, does not include correct number of samples'

test_set = torch.load("data/processed/test_loader.pth")
assert len(test_set) == 79, 'Test set, does not include correct number of samples'

data_train, label_train =  next(iter(train_set))
assert data_train.shape == torch.Size([64,28,28]), 'Train Data shape is incorrect'
assert label_train.shape == torch.Size([64]), 'Train Label shape is incorrect'
data_test, label_test =  next(iter(test_set))
assert data_test.shape == torch.Size([64,28,28]), 'Test Data shape is incorrect'
assert label_test.shape == torch.Size([64]), 'Test Label shape is incorrect'