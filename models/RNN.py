import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(num_features=100)
        self.dropout = nn.Dropout(p=0.2)
        self.rnn = torch.nn.RNN(input_size, hidden_size, nonlinearity='relu', batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.batch_norm(x)
        x, _ = self.rnn(x)
        x = self.linear(x)
        embedding = self.softmax(x)[:, -1, :]
        return embedding