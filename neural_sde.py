import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download("^NSEI", start="2000-01-01", end= "2025-12-31")
price = data["Close"].values.dropna()
returns = np.log(price/price.shift(1))
returns = returns.dropna()
lr = returns.values.flatten()

idx = int(0.8 * len(lr))
r_train = lr[:idx]
r_test = lr[idx:]
dt = 1/252
N = len(r_train)

class NeuralSDE(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(NeuralSDE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channels,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,out_channels),
        )
    def forward(self,x):
        out = self.net(x)
        out = nn.Softmax(out)
        return out



