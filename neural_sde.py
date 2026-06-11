import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

data = yf.download("^NSEI", start="2000-01-01", end= "2025-12-31")
dataVIX = yf.download("^INDIAVIX", start="2000-01-01", end= "2025-12-31")
price = data["Close"].values.dropna()
priceVIX = dataVIX["Close"].dropna().values
returnsVIX = np.log((priceVIX/priceVIX.shift(1)).dropna())
returns = np.log(price/price.shift(1))
returns = returns.dropna()
lr = returns.values.flatten()
lrVIX = returnsVIX.values.flatten()

idx = int(0.8 * len(lr))
r_train = lr[:idx]
r_test = lr[idx:]
dt = 1/252
N = len(r_train)

features = []
for i in range(len(lr)):
    rel_vol5 = lr.rolling(5).std()
    rel_vol20 = lr.rolling(20).std()
    mom_5 = lr.rolling(5).mean()
    mom_20 = lr.rolling(20).mean()
    r = lr[i]
    vix = lrVIX[i]
    vix_t = (lrVIX - lrVIX.shift(1))/lrVIX.shift(1)[i]
    skew = np.skew(lr)[i]
    kurt = np.kurtosis(lr)[i]
    f = [rel_vol5, rel_vol20, mom_5, mom_20, skew, kurt, vix_t, vix,r]
    features.append(f)

features = torch.tensor(features)
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



