
import argparse
import logging
import pdb
import sys

import hydra
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import torchdrift
from matplotlib import pyplot
import sklearn
from sklearn.manifold import Isomap

drift_detector = torchdrift.detectors.KernelMMDDriftDetector()
model = MyAwesomeModel()
model.load_state_dict(torch.load('models/checkpoint.pth'))

drift_detection_model = torch.nn.Sequential(
    model,
    drift_detector
)

trainloader = torch.load('data/processed/train_loader.pth')
testloader = torch.load('data/processed/test_loader.pth')
testloader_cr = torch.load('data/processed/corr_test_loader.pth')

image, label = next(iter(testloader))
print(image.shape)
img = image[0]
img = image.view(64,1,28,28) 
log_ps = model(img)
_, top_class = torch.exp(log_ps).topk(1, dim=1)
print(top_class, label[0])

torchdrift.utils.fit(trainloader, model, drift_detector)
score = drift_detector(log_ps)
p_val = drift_detector.compute_p_value(log_ps)
print(score, p_val)

N_base = drift_detector.base_outputs.size(0)
mapper = Isomap(n_components=2)
base_embedded = mapper.fit_transform(drift_detector.base_outputs)
features_embedded = mapper.transform(log_ps.detach().numpy())
pyplot.scatter(base_embedded[:, 0], base_embedded[:, 1], s=2, c='r')
pyplot.scatter(features_embedded[:, 0], features_embedded[:, 1], s=4)
pyplot.title(f'score {score:.2f} p-value {p_val:.2f}');
plt.savefig('drift')