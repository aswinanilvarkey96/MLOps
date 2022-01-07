import argparse
import logging
import pdb
import sys

import hydra
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import get_original_cwd
from model import MyAwesomeModel
from torch import nn, optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import os
print(os.getcwd())
from src.models.model import MyAwesomeModel

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
        preds, target = [], []
        cnt = 0
        for images, labels in testloader:  
            for idx,img in enumerate(images):    
                img = img.view(1,28,28) 
                log_ps = model(img)
                _, top_class = torch.exp(log_ps).topk(1, dim=1)
               # pdb.set_trace()
                log.info(f"Label: {labels[idx]}, Prediction: {top_class.item()}")
                if labels[idx]==top_class:
                    cnt+=1
                preds.append(log_ps.argmax(dim=-1))
            target.append(labels.detach())
            break
        target = torch.cat(target, dim=0)
        preds = torch.cat(preds, dim=0)
        report = classification_report(target, preds)
        with open("classification_report.txt", 'w') as outfile:
            outfile.write(report)
        confmat = confusion_matrix(target, preds)
        disp = ConfusionMatrixDisplay( confmat )
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        plt.figure()
        plt.plot(range(1,10))
        plt.show()
        
       # pdb.set_trace()
        log.info(f'Accuracy: {cnt/idx*100:.2f}%')
    model.train()

if __name__ == "__main__":
    predict()
