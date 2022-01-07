import torch
import torch.nn.functional as F
from torch import nn, optim




class MyAwesomeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Inputs to hidden layer linear transformation
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        # Output layer, 10 units - one for each digit
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)
        


    def forward(self, x):
        # Hidden layer with sigmoid activation
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # Output layer with softmax activation
        x = F.log_softmax(self.fc4(x), dim=1)

        return x
    
    
