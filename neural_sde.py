import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import skew, kurtosis

data = yf.download("^NSEI", start="2000-01-01", end= "2025-12-31")
dataVIX = yf.download("^INDIAVIX", start="2000-01-01", end= "2025-12-31")
price = data["Close"].dropna()
priceVIX = dataVIX["Close"].dropna()
returnsVIX = np.log((priceVIX/priceVIX.shift(1)).dropna())
returns = np.log(price/price.shift(1))
returns = returns.dropna()
lr = returns
rel_vol5 = lr.rolling(5).std()
rel_vol20 = lr.rolling(20).std()
mom_5 = lr.rolling(5).mean()
mom_20 = lr.rolling(20).mean()
lr = returns.values.flatten()
lrVIX = returnsVIX.values.flatten()

idx = int(0.8 * len(lr))
r_train = lr[:idx]
r_test = lr[idx:]
dt = 1/252
N = len(r_train)
features = []
for i in range(len(lr)):
    rel_5 = rel_vol5[i]
    rel_20 = rel_vol20[i]
    m5 = mom_5[i]
    m20 = mom_20[i]
    r = lr[i]
    vix = lrVIX[i]
    if(i == 0):
        vix_t = 0
    else:
        vix_t = (lrVIX[i] - lrVIX[i - 1]) / (lrVIX[i - 1] + 1e-8)
    skewi = skew(lr[max(0, i - 20):i]) if i > 1 else 0
    kurti = kurtosis(lr[max(0, i - 20):i]) if i > 1 else 0.0
    f = [rel_vol5.iloc[i], rel_vol20.iloc[i], mom_5.iloc[i], mom_20.iloc[i], skewi, kurti, vix_t, vix,r]
    features.append(f)
features = features[20:]
features = torch.tensor(features)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuralSDE(nn.Module):
    def __init__(self,feature_dim,latent_dim):
        super(NeuralSDE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.Tanh(),
            nn.Linear(64, latent_dim)
        )
        self.drift = MLP(feature_dim+latent_dim,latent_dim)
        self.diff = MLP(feature_dim+latent_dim,latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Tanh(),
            nn.Linear(64,1)
        )
    def forward(self,x,h = None):
        if h is None:
            h = self.encoder(x)
        inp = torch.cat([h, x], dim=-1)
        mu = self.drift(inp)
        sigma = F.softplus(self.diff(inp))
        dw = torch.randn_like(h) * np.sqrt(dt)
        h_new = h + mu * dt + sigma * dw
        out = self.decoder(h_new)
        return out, h_new

model = NeuralSDE(9,64)
optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
criterion = torch.nn.MSELoss()
X = features[:idx]
y = torch.tensor(lr[20:idx+20], dtype=torch.float32).unsqueeze(1)
h = None
for epoch in range(200):
    optimizer.zero_grad()
    out, h = model(X,h)
    h.detach()
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(loss.item())






