import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize

data = yf.download("^NSEI", start="2000-01-01", end= "2025-12-31")
prices = data["Close"].dropna()
print(data.head())
print(data.shape)
returns = data["Close"]/data["Close"].shift(1)
log_returns = np.log(returns).dropna() * 100

lr = log_returns.values.flatten()
print(f"Mean log return: {np.mean(lr)}")
print(f"Standard deviation: {np.std(lr)}")
print(f"Skewness: {stats.skew(lr)}")
print(f"Kurtosis: {stats.kurtosis(lr)}")

idx = int(len(lr)*0.8)
r_train = lr[:idx]
r_test = lr[idx:]

def garch_filter(parameters, lr):
    omega, alpha, beta, mu = parameters
    n = len(lr)
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = np.var(lr)
    eps[0]    = lr[0] - mu
    for i in range(1,n):
        sigma2[i] = beta*sigma2[i-1] + alpha * eps[i-1]**2 + omega
        eps[i] = lr[i] - mu
    return sigma2, eps

def likelihood(parameters,lr):
    omega, alpha, beta, mu = parameters
    if omega <= 0 or alpha<0 or beta<0 or alpha+beta>1:
        return 1e10
    sigma2 , eps = garch_filter(parameters,lr)

    ll = -0.5 * np.sum(np.log(2*np.pi) + np.log(sigma2) + eps**2/sigma2)
    return -ll

def fit(lr):
    x0 = [np.var(lr) * 0.05, 0.05, 0.85, np.mean(lr)]
    result = minimize(likelihood,x0,args=(lr,),method='L-BFGS-B',options={'maxiter': 1000})
    return result.x

params = fit(r_train)
omega, alpha, beta, mu = params
print(f"omega = {omega:.6f}")
print(f"alpha = {alpha:.6f}")
print(f"beta  = {beta:.6f}")
print(f"alpha + beta = {alpha + beta:.6f}")
def forecast(parameters, lr, h= 22):
    omega, alpha, beta, mu = parameters
    sigma2, eps = garch_filter(parameters,lr)
    sigma2T = sigma2[-1]
    epsT = eps[-1]
    lr_var = omega/(1-alpha-beta)
    forecasts = np.zeros(h)
    forecasts[0] = omega + alpha * epsT**2 + beta * sigma2T
    for i in range(1,h):
        forecasts[i] = lr_var + (alpha+beta)**i*(forecasts[0] - lr_var)
    return np.sqrt(forecasts)

vol_fc = forecast(params, r_train, h=10000)
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
print(f"10000-day forecast: {vol_fc[9999]:.4f}%")

f = vol_fc**2
plt.plot(f)
plt.axhline(
    omega/(1-alpha-beta),
    color='red',
    linestyle='--',
    label='Unconditional Variance'
)

plt.legend()
plt.title("Convergence of Conditional Variance Forecast")
plt.savefig("Convergence of Conditional Variance Forecast in GARCH.png")

"""
1-day  forecast: 1.2140%
5-day  forecast: 1.2255%
22-day forecast: 1.2705%
50-day forecast: 1.3326%
100-day forecast: 1.4159%
200-day forecast: 1.5164%
300-day forecast: 1.5686%
400-day forecast: 1.5965%
600-day forecast: 1.6197%
1000-day forecast: 1.6287%
3000-day forecast: 1.6296%
10000-day forecast: 1.6296%
"""


