import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim

from src.models.model import MyAwesomeModel

import hydra
from hydra.utils import get_original_cwd

import logging
log = logging.getLogger(__name__)
@hydra.main(config_path = '../../src/config/' , config_name="testconfig")

def predict(cfg):
    testpath = cfg.hyperparameters.testpath
    checkpoint = cfg.hyperparameters.checkpoint
    
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(str(get_original_cwd()) + checkpoint))
    testloader = torch.load(str(get_original_cwd()) +testpath)
    model.eval()
    with torch.no_grad():
        cnt = 0
        for images, labels in testloader:  
            for idx,img in enumerate(images):    
                img = img.view(1,28,28) 
                log_ps = model(img)
                _, top_class = torch.exp(log_ps).topk(1, dim=1)
                log.info(f"Label: {labels[idx]}, Prediction: {top_class.item()}")
                if labels[idx]==top_class:
                    cnt+=1
            break
        log.info(f'Accuracy: {cnt/idx*100:.2f}%')
    model.train()

if __name__ == "__main__":
    predict()
