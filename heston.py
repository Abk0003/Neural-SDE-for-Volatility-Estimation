import numpy as np
import pandas as pdb
import yfinance as yf
import matplotlib.pyplot as plt

data = yf.download("^NSEI", start="2000-01-01", end= "2025-12-31")
price = data["Close"].dropna()
print(data.head())
print(data.shape)
returns = np.log(price/price.shift(1))
returns = returns.dropna()

lr = returns.values.flatten()

idx = int(0.8 * len(lr))
r_train = lr[:idx]
r_test = lr[idx:]
dt = 1/252
N = len(r_train)

mu      = np.mean(r_train) * 252
kappa   = 2.0
theta   = np.var(lr) * 252
epsilon = 0.5
rho     = -0.7
S = np.zeros(N)
v = np.zeros(N)
S[0] = float(price.iloc[0,0])
v[0] = np.var(lr) *252
#EULER METHOD
"""
for i in range(1,N):
        Z1 = np.random.normal(0, 1)
        Z3 = np.random.normal(0, 1)
        Z2 = Z1 * rho + (1 - rho ** 2) ** (0.5) * Z3
        v[i] = v[i-1]*(1 - kappa*dt) + kappa*theta*dt + epsilon*(max(v[i-1],1e-12))**0.5 * dt**(0.5) * Z2
        S[i] = S[i-1]*(1 + mu*dt + max(v[i-1],1e-12)**(0.5) * dt**(0.5) * Z1)

plt.plot(S)
plt.savefig("hestonS.png")
plt.show()

plt.plot(v)
plt.savefig("hestonV.png")
plt.show()"""

#QE METHOD
for i in range(1,N):
        m = v[i-1]*np.exp(-kappa*dt) + theta*(1-np.exp(-kappa*dt))
        s2 = epsilon**2/kappa * v[i-1]*np.exp(-kappa*dt) *(1-np.exp(-kappa*dt)) + theta*epsilon**2/(2*kappa) * (1-np.exp(-kappa*dt))**2
        phi = s2/m**2
        Zv = np.random.normal(0, 1)
        Zp = np.random.normal(0, 1)
        Zs = Zv * rho + (1 - rho ** 2) ** (0.5) * Zp
        if phi <= 1.5:
                b2 = 2 / phi - 1 + np.sqrt(2 / phi * (2 / phi - 1))
                a = m / (1 + b2)
                v[i] = max(a * (np.sqrt(b2) + Zv) ** 2, 1e-10)
        else:
                p = (phi - 1) / (phi + 1)
                beta = (1 - p) / m
                U = np.random.uniform(0, 1)
                v[i] = max(0.0 if U <= p else np.log((1 - p) / (1 - U)) / beta, 1e-10)
        #S[i] = S[i - 1] * (1 + mu * dt + max(v[i], 1e-12) ** (0.5) * dt ** (0.5) * Zs) [Aritemetic]
        S[i] = S[i - 1] * np.exp((mu - 0.5 * v[i-1]) * dt + np.sqrt(v[i-1] * dt) * Zs) #[geometric]

plt.plot(S)
plt.savefig("hestonSQE.png")
plt.show()

plt.plot(v)
plt.savefig("hestonVQE.png")
plt.show()
vol_fc = v**0.5
print(f"1-day  forecast: {vol_fc[0] :.4f}%")
print(f"5-day  forecast: {vol_fc[4]:.4f}%")
print(f"22-day forecast: {vol_fc[21]:.4f}%")
print(f"50-day forecast: {vol_fc[49]:.4f}%")
print(f"100-day forecast: {vol_fc[99]:.4f}%")
print(f"200-day forecast: {vol_fc[199]:.4f}%")
print(f"300-day forecast: {vol_fc[299]:.4f}%")
print(f"400-day forecast: {vol_fc[399]:.4f}%")
print(f"600-day forecast: {vol_fc[599]:.4f}%")
print(f"1000-day forecast: {vol_fc[999]:.4f}%")
print(f"3000-day forecast: {vol_fc[2999]:.4f}%")

"""
1-day  forecast: 0.0131%
5-day  forecast: 0.0012%
22-day forecast: 0.0002%
50-day forecast: 0.0002%
100-day forecast: 0.0003%
200-day forecast: 0.0017%
300-day forecast: 0.0019%
400-day forecast: 0.0018%
600-day forecast: 0.0001%
1000-day forecast: nan%
3000-day forecast: nan%"""








