import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import skew, kurtosis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")

nifty = yf.download("^NSEI", start="2000-01-01", end="2025-12-31")
vix = yf.download("^INDIAVIX", start="2000-01-01", end="2025-12-31")
df = pd.DataFrame()
df["ret"] = np.log(nifty["Close"]/nifty["Close"].shift(1))
df["ret_v"] = np.log(vix["Close"]/vix["Close"].shift(1))
df["mom_5"] = df["ret"].rolling(5).mean().shift(1)
df["mom_20"] = df["ret"].rolling(20).mean().shift(1)
df["vol_5"] = df["ret"].rolling(5).std().shift(1)
df["vol_20"] = df["ret"].rolling(20).std().shift(1)
df["vov20"] = df["ret_v"].rolling(20).std().shift(1)
df["vov60"] = df["ret_v"].rolling(60).std().shift(1)
df["ret_lag"] = df["ret"].shift(1)
df["vix_lag"] = df["ret_v"].shift(1)
df["skew20"] = df["ret"].rolling(20).apply(skew, raw=True).shift(1)
df["kurt20"] = df["ret"].rolling(20).apply(kurtosis, raw=True).shift(1)
df["spread"] = (nifty["High"] - nifty["Low"]).shift(1)
df["vix_t"]   = df["ret_v"].diff() / (df["ret_v"].shift(1).abs() + 1e-8)
df["target"] =  df["ret"].rolling(5).std().shift(-5)
df = df.dropna()
dt = 1/252
features_col = ["mom_5","mom_20","vol_5","vol_20","vov20","vov60","ret_lag","vix_lag","skew20","kurt20","spread","vix_t"]
features = df[features_col]
features = features.values
y = df["target"].values
features = (features - features.mean()) / (features.std() + 1e-8)
features = torch.tensor(features, dtype=torch.float32).to(device)
y = (y - y.mean()) / (y.std() + 1e-8)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
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
            nn.Linear(64,1),
        )
    def forward(self,x,h = None):
        if h is None:
            h = self.encoder(x)
        inp = torch.cat([h, x], dim=-1)
        mu = self.drift(inp)
        sigma = F.softplus(self.diff(inp))
        dw = torch.randn_like(h) * np.sqrt(dt) * 0.1
        h_new = h + mu * dt + sigma * dw
        out = self.decoder(h_new)
        return out, h_new

idx = int(0.8 * len(features))
X_train = features[:idx].to(device)
X_test = features[idx:].to(device)
y_train = y[:idx].to(device)
y_test = y[idx:].to(device)
model = NeuralSDE(12,8).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
criterion = torch.nn.MSELoss()

for epoch in range(200):
    epoch_loss = 0.0
    h = None
    model.train()

    for t in range(len(X_train)):
        out, h = model(X_train[t:t+1], h)
        loss = criterion(out, y_train[t:t+1])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        h = h.detach()
        epoch_loss += loss.item()

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        h_val = None
        for t in range(len(X_test)):
            out, h_val = model(X_test[t:t + 1], h_val)
            val_loss += criterion(out, y_test[t:t + 1]).item()

    print(f"Epoch {epoch:3d} | train: {epoch_loss / len(X_train):.5f} | val: {val_loss / len(X_test):.5f}")







