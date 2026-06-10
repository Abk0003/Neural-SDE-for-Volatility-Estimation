import yfinance as yf
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data = yf.download("^VIX", start = "2000-01-01", end = "2026-01-01")
log_return = np.log(data["Close"]/data["Close"].shift(1))
log_return = log_return.dropna()
lr = log_return.values.flatten()
print(f"Mean log return: {np.mean(lr)}")
print(f"Standard deviation: {np.std(lr)}")
print(f"Skewness: {stats.skew(lr)}")
print(f"Kurtosis: {stats.kurtosis(lr)}")

jb_stat, jb_p = stats.jarque_bera(lr)
print(f"Jarque-Bera statistic: {jb_stat}")
print(f"Jarque-Bera p-value: {jb_p}")

z = (lr - lr.mean())/lr.std()
normal_99 = stats.norm.ppf(0.01)
actual_99 = np.percentile(z,1)
if abs(actual_99) > abs(normal_99):
    print(f"Normal: {abs(normal_99)} vs Actual: {abs(actual_99)} \nClearly, fat tails exist")
else:
    print(f"Normal{abs(normal_99)} vs Actual{abs(actual_99)}. \nClearly, thin tails exist")

extreme_pct = np.mean(np.abs(z) > 3) * 100
normal_pct = (1-stats.norm.cdf(3))*2*100
print(f"  Normal predicts: {normal_pct:.4f}%")
print(f"  Empirical:       {extreme_pct:.4f}%  ({extreme_pct/normal_pct:.1f}x more frequent)")

rolling_vol = log_return.rolling(window=30).std() * np.sqrt(252)
lb_result = acorr_ljungbox(lr**2, lags=10, return_df=True)
print(f"p-values (lags 1-10): {lb_result['lb_pvalue'].values.round(4)}")









