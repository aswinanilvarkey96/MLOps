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
    model = torch.load("models/checkpoint.pth")

    print("Evaluating until hitting the ceiling")
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    with torch.no_grad():
        cnt = []
        running_loss = 0
        for images, labels in testloader:
            log_ps = model(images)
            loss = criterion(log_ps, labels.long())
            running_loss += loss.item()
            _, top_class = torch.exp(log_ps).topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            cnt.append(torch.mean(equals.type(torch.FloatTensor)))
        acc = torch.sum(torch.tensor(cnt)) / len(cnt)
        print(f"Accuracy: {acc*100:.2f}%")

    torch.save(NN.state_dict(), "models/checkpoint.pth")


if __name__ == "__main__":
    test()
