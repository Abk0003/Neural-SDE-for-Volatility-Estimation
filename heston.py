import numpy as np
import pandas as pdb
import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download("^NSEI", start="2000-01-01", end= "2025-12-31")
price = data["Close"].dropna()
returns = np.log(price/price.shift(1))
returns = returns.dropna()

lr = returns.values.flatten()

idx = int(0.8 * len(lr))
r_train = lr[:idx]
r_test = lr[idx:]
dt = 1/252
N = len(r_train)
mu      = 0.08
kappa   = 2.0
theta   = np.var(lr)
epsilon = 0.3
rho     = -0.7
S = np.zeros(N)
v = np.zeros(N)
S[0] = float(price.iloc[0,0])
v[0] = np.var(lr)
for i in range(1,N):
        Z1 = np.random.normal(0, 1)
        Z3 = np.random.normal(0, 1)
        Z2 = Z1 * rho + (1 - rho ** 2) ** (0.5) * Z3
        v[i] = v[i-1]*(1 - kappa*dt) + kappa*theta*dt + epsilon*(max(v[i-1],0))**0.5 * dt**(0.5) * Z2
        S[i] = S[i-1]*(1 + mu*dt + max(v[i-1],0)**(0.5) * dt**(0.5) * Z1)

plt.plot(S)
plt.show()

plt.plot(v)
plt.show()








