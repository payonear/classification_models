import torch
import torch.nn as nn
from torch.nn import functional as F

class Net(nn.Module):
    """Neural network which can be used for classification or regression tasks. As an input takes 
        vectorised text embeddings. Outputs predicted probability distribution and loss value.

    Args:
        input_dim: input dimension
        output_dim: output dimension
        loss_func: loss function (e.g. torch.nn.CrossEntropy())
        hidden_dim: hidden dimension. Default value - 128.
        dropout: dropout rate in range from 0 to 1. Default value 0.0.
    
    Output:
        tuple of predicted probability distribution and loss value.
    """
    def __init__(self, input_dim, output_dim, loss_func, hidden_dim=128, dropout=0.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.loss_func = loss_func

    def forward(self, vector, label=None):
        layer1 = self.dropout(F.relu(self.fc1(vector)))
        pred = self.fc2(layer1)
        loss = self.loss_func(pred, label) if label is not None else None
        return(F.softmax(pred, dim=-1), loss)
