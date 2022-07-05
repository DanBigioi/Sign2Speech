import torch.nn.functional as F
import pytorch_lightning as pl
import torch
from typing import Tuple
from torch import nn


class lstm(nn.Module):
    def __init__(self, input_size: int = 63, hidden_size: int = 256, num_layers: int = 4, bidirectional = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.bilstm = nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  num_layers=num_layers,
                                  dropout=0.1,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        self.fc_in_features = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Sequential(
            nn.Linear(in_features=self.fc_in_features, out_features=256),
            nn.BatchNorm1d(101),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.Dropout(p=0.5),
            nn.BatchNorm1d(101),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),

        )

    def forward(self, X):
        lstm_out, (hn, cn) = self.bilstm(X)
        return self.fc(lstm_out)