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
df["spread"] = ((nifty["High"] - nifty["Low"])/nifty["Close"]).shift(1)
df["vix_t"]   = df["ret_v"].diff() / (df["ret_v"].shift(1).abs() + 1e-8)
df["target"] =  df["ret"].rolling(5).std().shift(-5)
df = df.dropna()
dt = 1/252
features_col = ["mom_5","mom_20","vol_5","vol_20","vov20","vov60","ret_lag","vix_lag","skew20","kurt20","spread","vix_t"]
features = df[features_col]
features = features.values
y = df["target"].values
features = torch.tensor(features, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, out_dim)
        self.drop = nn.Dropout(p=0.1)
    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = self.drop(x)
        x = F.tanh(self.fc2(x))
        x = self.drop(x)
        x = self.fc3(x)
        return x

class NeuralSDE(nn.Module):
    def __init__(self,feature_dim,latent_dim):
        super(NeuralSDE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.Tanh(),
            nn.Linear(16, latent_dim)
        )
        self.drift = MLP(feature_dim+latent_dim,latent_dim)
        self.diff = MLP(feature_dim+latent_dim,latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.Tanh(),
            nn.Linear(16,1),
        )
    def forward(self,x,h = None):
        if h is None:
            h = self.encoder(x)
        inp = torch.cat([h, x], dim=-1)
        mu = self.drift(inp)
        sigma = 0.1*F.softplus(self.diff(inp))
        dw = torch.randn_like(h) * np.sqrt(dt) * 0.01
        h_new = h + mu * dt + sigma * dw
        out = self.decoder(h_new)
        return out, h_new

idx = int(0.8 * len(features))
X_train = features[:idx].to(device)
X_test = features[idx:].to(device)
y_train = y[:idx].to(device)
y_test = y[idx:].to(device)
x_mean = X_train.mean(0, keepdim=True)
x_std  = X_train.std(0, keepdim=True)

X_train = (X_train - x_mean)/x_std
X_test  = (X_test  - x_mean)/x_std

y_mean = y_train.mean()
y_std  = y_train.std()

y_train = (y_train-y_mean)/y_std
y_test  = (y_test-y_mean)/y_std
model = NeuralSDE(12,8).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=3e-4,weight_decay=1e-4)
criterion = torch.nn.MSELoss()

seq_len = 60

for epoch in range(200):
    h = None
    epoch_loss = 0

    for start in range(0, len(X_train)-seq_len):

        optimizer.zero_grad()

        loss = 0

        for t in range(start, start+seq_len):
            out, h = model(X_train[t:t+1], h)
            loss += criterion(out, y_train[t:t+1])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
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







