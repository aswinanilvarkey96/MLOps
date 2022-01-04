import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim

from src.models.model import MyAwesomeModel


def test():
    testloader = torch.load("data/processed/test_loader.pth")

    print("Training day and night")
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.1)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    # Implement training loop here
    NN = MyAwesomeModel()
    model = torch.load("models/checkpoint1.pth")

    print("Evaluating until hitting the ceiling")
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    with torch.no_grad():
        cnt = []
        running_loss = 0
        i = 0
        for images, labels in testloader:
            log_ps = model(images)
            loss = criterion(log_ps, labels.long())
            running_loss += loss.item()
            _, top_class = torch.exp(log_ps).topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            cnt.append(torch.mean(equals.type(torch.FloatTensor)))
            accuracy = torch.mean(equals.type(torch.FloatTensor))
            print(f'Accuracy: {accuracy.item()*100}%')
            i+=1
        acc = torch.sum(torch.tensor(cnt)) / len(cnt)
        print(f"Accuracy: {acc*100:.2f}%")

    torch.save(NN.state_dict(), "models/checkpoint.pth")


def predict():
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--model_path', default="models/checkpoint.pth")
    parser.add_argument('--folder_path',default="data/processed/test_1.pth")
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)
    testloader = torch.load("data/processed/test_loader.pth")
    model = MyAwesomeModel()
    model.load_state_dict(torch.load("models/checkpoint1.pth"))
    with torch.no_grad():
        cnt = 0
        for images, labels in testloader:  
            for idx,img in enumerate(images):    
                img = img.view(1,28,28) 
                log_ps = model(img)
                _, top_class = torch.exp(log_ps).topk(1, dim=1)
                print(f"Label: {labels[idx]}, Prediction: {top_class.item()}")
                if labels[idx]==top_class:
                    cnt+=1
            break
        print(f'Accuracy: {cnt/idx*100:.2f}%')

if __name__ == "__main__":
    predict()
