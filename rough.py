
"""
Neural SDE v2 — Nifty 5-day Realized Vol Forecast
===================================================
Key fixes over v1:
  1. GRU-gated latent update   → proper temporal memory, no vanishing gradient
  2. Noise off at eval         → train/val use consistent signal (no stochastic mismatch)
  3. Persistent h across epochs→ model sees continuous temporal history, not a reset each epoch
  4. Huber loss (delta=1.5)    → robust to vol spike outliers vs MSE
  5. Warmup + cosine LR        → avoids large early updates blowing up latent state
  6. Best-checkpoint restore   → returns weights at lowest val loss, not last epoch
  7. MC rollout uncertainty    → 30 stochastic forward passes give epistemic error bars
  8. Log-vol target            → standardised Gaussian-ish target; back-transform with exp()
  9. Extra features            → vol_60, ret_lag2, leverage effect, vol-ratio
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import skew, kurtosis

# ── Config ────────────────────────────────────────────────────────────────────
SEQ_LEN    = 120    # TBPTT unroll; longer = better vol-persistence capture
LATENT     = 8    # latent state dimension
HIDDEN     = 16  # MLP hidden width
EPOCHS     = 200
LR         = 5e-4
CLIP_NORM  = 1.0
DROPOUT    = 0.15
MC_SAMPLES = 30     # stochastic forward passes for uncertainty
dt         = 1 / 252

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Data ──────────────────────────────────────────────────────────────────────
print("Downloading data ...")
nifty = yf.download("^NSEI",     start="2000-01-01", end="2025-12-31", progress=False)
vix   = yf.download("^INDIAVIX", start="2000-01-01", end="2025-12-31", progress=False)

# Handle MultiIndex columns (newer yfinance)
if isinstance(nifty.columns, pd.MultiIndex):
    nifty.columns = nifty.columns.get_level_values(0)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)

df = pd.DataFrame(index=nifty.index)
df["ret"]    = np.log(nifty["Close"] / nifty["Close"].shift(1))
df["ret_v"]  = np.log(vix["Close"]   / vix["Close"].shift(1))

df["mom_5"]      = df["ret"].rolling(5).mean().shift(1)
df["mom_20"]     = df["ret"].rolling(20).mean().shift(1)
df["vol_5"]      = df["ret"].rolling(5).std().shift(1)
df["vol_20"]     = df["ret"].rolling(20).std().shift(1)
df["vol_60"]     = df["ret"].rolling(60).std().shift(1)   # long-run vol anchor
df["vov20"]      = df["ret_v"].rolling(20).std().shift(1)
df["vov60"]      = df["ret_v"].rolling(60).std().shift(1)
df["ret_lag1"]   = df["ret"].shift(1)
df["ret_lag2"]   = df["ret"].shift(2)
df["vix_lag"]    = df["ret_v"].shift(1)
df["skew20"]     = df["ret"].rolling(20).apply(skew,     raw=True).shift(1)
df["kurt20"]     = df["ret"].rolling(20).apply(kurtosis, raw=True).shift(1)
df["spread"]     = ((nifty["High"] - nifty["Low"]) / nifty["Close"]).shift(1)
df["vix_t"]      = df["ret_v"].diff() / (df["ret_v"].shift(1).abs() + 1e-8)
# Leverage effect: negative return predicts vol increase
df["lev_effect"] = (df["ret"].shift(1) * df["vol_5"]).shift(1)
# Vol ratio: how stretched current vol is vs long-run (mean-reversion signal)
df["vol_ratio"]  = (df["vol_5"] / (df["vol_60"] + 1e-8)).shift(1)

df["target_raw"] = df["ret"].rolling(5).std().shift(-5)
df = df.dropna()
print(f"Samples after dropna: {len(df)}")

FEATURE_COLS = [
    "mom_5", "mom_20", "vol_5", "vol_20", "vol_60",
    "vov20", "vov60", "ret_lag1", "ret_lag2", "vix_lag",
    "skew20", "kurt20", "spread", "vix_t", "lev_effect", "vol_ratio"
]
FEAT_DIM = len(FEATURE_COLS)

# ── Split ─────────────────────────────────────────────────────────────────────
split    = int(0.8 * len(df))
tr, te   = df.iloc[:split], df.iloc[split:]

X_tr     = tr[FEATURE_COLS].values.astype(np.float32)
X_te     = te[FEATURE_COLS].values.astype(np.float32)

x_mean   = X_tr.mean(0, keepdims=True)
x_std    = X_tr.std(0,  keepdims=True) + 1e-8
X_tr     = (X_tr - x_mean) / x_std
X_te     = (X_te - x_mean) / x_std

log_y_tr = np.log(tr["target_raw"].values.astype(np.float32) + 1e-8)
log_y_te = np.log(te["target_raw"].values.astype(np.float32) + 1e-8)
y_mean, y_std = log_y_tr.mean(), log_y_tr.std() + 1e-8
log_y_tr = (log_y_tr - y_mean) / y_std
log_y_te = (log_y_te - y_mean) / y_std

X_train  = torch.tensor(X_tr).to(device)
X_test   = torch.tensor(X_te).to(device)
y_train  = torch.tensor(log_y_tr).unsqueeze(1).to(device)
y_test   = torch.tensor(log_y_te).unsqueeze(1).to(device)
print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# ── Model ─────────────────────────────────────────────────────────────────────
class GatedNeuralSDE(nn.Module):
    """
    Latent SDE with GRU-gated Euler-Maruyama update:

        z     = σ( W_z · [h, x] )            # how much to update
        h̃    = tanh( W_c · [h, x] )         # candidate new state
        σ(·)  = 0.02 · softplus( MLP([h,x]) )# diffusion coefficient
        dW    = N(0,1) · √dt                  # Brownian increment (train only)

        h_new = LayerNorm( (1−z)·h + z·h̃ + σ·dW )

    Why this works better than v1 plain MLP:
      • Gate z controls memory retention — model learns how much past to keep
      • (1−z)·h term is identity-like shortcut → no vanishing gradient
      • LayerNorm prevents latent magnitude explosion over long sequences
      • SDE noise is switched off at eval → deterministic mean prediction;
        MC rollout gives uncertainty by re-enabling noise at inference.
    """
    def __init__(self, feat_dim, latent_dim, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        inp = feat_dim + latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
        # GRU-style gating
        self.update_gate   = nn.Linear(inp, latent_dim)
        self.candidate_net = nn.Linear(inp, latent_dim)
        # Diffusion σ(h, x)
        self.diff_net = nn.Sequential(
            nn.Linear(inp, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, latent_dim),
        )
        self.ln = nn.LayerNorm(latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def init_hidden(self, x0):
        return self.encoder(x0)                            # [1, L]

    def step(self, h, x, add_noise=True):
        inp    = torch.cat([h, x], dim=-1)
        z      = torch.sigmoid(self.update_gate(inp))     # [1, L]
        h_cand = torch.tanh(self.candidate_net(inp))      # [1, L]
        sigma  = 0.02 * F.softplus(self.diff_net(inp))   # [1, L]
        dW     = torch.randn_like(h) * (dt ** 0.5) if add_noise else torch.zeros_like(h)
        h_new  = (1 - z) * h + z * h_cand + sigma * dW
        return self.ln(h_new)

    def forward(self, X_seq, add_noise=False):
        """X_seq: [T, F]  →  preds: [T, 1], h_final: [1, L]"""
        h = self.init_hidden(X_seq[0:1])
        preds = []
        for t in range(X_seq.shape[0]):
            h = self.step(h, X_seq[t:t+1], add_noise)
            preds.append(self.decoder(h))
        return torch.cat(preds, dim=0), h

    @torch.no_grad()
    def predict_mc(self, X_seq, n_samples=MC_SAMPLES):
        """MC rollout with noise enabled → epistemic uncertainty."""
        self.eval()
        runs = []
        for _ in range(n_samples):
            preds, _ = self.forward(X_seq, add_noise=True)
            runs.append(preds)
        runs = torch.stack(runs, dim=0)        # [S, T, 1]
        return runs.mean(0), runs.std(0)       # [T, 1], [T, 1]


# ── Training ──────────────────────────────────────────────────────────────────
model     = GatedNeuralSDE(FEAT_DIM, LATENT, HIDDEN).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-4)

def lr_lambda(epoch):
    warmup = 10
    if epoch < warmup:
        return epoch / warmup                  # linear warmup
    progress = (epoch - warmup) / max(EPOCHS - warmup, 1)
    return 0.5 * (1.0 + np.cos(np.pi * progress))  # cosine decay

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

T_train = X_train.shape[0]
train_losses, val_losses = [], []
best_val, best_state     = float("inf"), None

# KEY: persistent hidden state — model sees unbroken history across epochs
h_persistent = model.init_hidden(X_train[0:1]).detach()

print(f"\nTraining {EPOCHS} epochs | TBPTT chunk={SEQ_LEN} | latent={LATENT}\n")

for epoch in range(EPOCHS):
    model.train()

    h          = h_persistent          # carry state from last epoch
    epoch_loss = 0.0
    n_chunks   = 0

    for start in range(0, T_train - SEQ_LEN, SEQ_LEN):   # non-overlapping
        end     = min(start + SEQ_LEN, T_train)
        X_chunk = X_train[start:end]
        y_chunk = y_train[start:end]

        optimizer.zero_grad()
        preds = []
        for t in range(len(X_chunk)):
            h = model.step(h, X_chunk[t:t+1], add_noise=True)
            preds.append(model.decoder(h))

        preds = torch.cat(preds, dim=0)
        loss  = F.huber_loss(preds, y_chunk, delta=1.5)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
        optimizer.step()

        h = h.detach()
        epoch_loss += loss.item()
        n_chunks   += 1

    # Save end-of-epoch h for next epoch's warm start
    h_persistent = h.detach()
    scheduler.step()

    # ── Validation: deterministic (no noise) ──────────────────────────────
    model.eval()
    with torch.no_grad():
        val_preds, _ = model(X_test, add_noise=False)
        avg_val      = F.huber_loss(val_preds, y_test, delta=1.5).item()

    avg_tr = epoch_loss / max(n_chunks, 1)
    train_losses.append(avg_tr)
    val_losses.append(avg_val)

    if avg_val < best_val:
        best_val   = avg_val
        best_state = {k: v.clone() for k, v in model.state_dict().items()}


    print(f"Epoch {epoch:3d}/{EPOCHS}  train {avg_tr:.4f}  val {avg_val:.4f}  "
        f"lr {scheduler.get_last_lr()[0]:.2e}  best {best_val:.4f}")

# ── Restore best checkpoint ───────────────────────────────────────────────────
model.load_state_dict(best_state)
print(f"\nRestored best checkpoint (val={best_val:.4f})")

# ── MC inference ──────────────────────────────────────────────────────────────
print(f"Running {MC_SAMPLES} MC rollouts ...")
pred_mean, pred_std = model.predict_mc(X_test, n_samples=MC_SAMPLES)

pred_mean_np = pred_mean.cpu().numpy().squeeze()
pred_std_np  = pred_std.cpu().numpy().squeeze()

# Back-transform: normalised log-vol → raw vol
pred_logvol  = pred_mean_np * y_std + y_mean
pred_vol     = np.exp(pred_logvol)
# Uncertainty in vol space via delta method: Δvol ≈ vol * Δlog_vol
pred_unc_vol = pred_vol * (pred_std_np * y_std)

true_vol     = te["target_raw"].values

mae  = np.mean(np.abs(pred_vol - true_vol))
rmse = np.sqrt(np.mean((pred_vol - true_vol) ** 2))
corr = np.corrcoef(pred_vol, true_vol)[0, 1]
print(f"\n── Test (raw vol) ──  MAE={mae:.5f}  RMSE={rmse:.5f}  Corr={corr:.4f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1. Loss curves
ax = axes[0]
ax.plot(train_losses, label="Train (Huber)", lw=1.5)
ax.plot(val_losses,   label="Val  (Huber)",  lw=1.5)
ax.axhline(best_val, color="red", ls="--", lw=1, alpha=0.7,
           label=f"Best val={best_val:.4f}")
ax.set(title="Loss curves (Huber, normalised log-vol)",
       xlabel="Epoch", ylabel="Huber loss")
ax.legend(); ax.grid(alpha=0.3)

# 2. Forecast + MC uncertainty band
ax = axes[1]
dates = te.index
ax.fill_between(dates,
                np.maximum(pred_vol - pred_unc_vol, 0),
                pred_vol + pred_unc_vol,
                alpha=0.25, color="C1", label=f"±1σ MC ({MC_SAMPLES} runs)")
ax.plot(dates, true_vol, lw=1.2, color="C0", label="True vol",   alpha=0.9)
ax.plot(dates, pred_vol, lw=1.2, color="C1", label="Pred vol",   alpha=0.9)
ax.set(title=f"5-day realized vol forecast  (Corr={corr:.3f}  RMSE={rmse:.5f})",
       xlabel="Date", ylabel="Volatility")
ax.legend(); ax.grid(alpha=0.3)

# 3. Scatter: predicted vs true
ax = axes[2]
lim = max(true_vol.max(), pred_vol.max()) * 1.05
ax.scatter(true_vol, pred_vol, alpha=0.3, s=8, color="steelblue")
ax.plot([0, lim], [0, lim], "r--", lw=1.5, label="Perfect forecast")
ax.set(title="Predicted vs True (test set)",
       xlabel="True vol", ylabel="Predicted vol",
       xlim=(0, lim), ylim=(0, lim))
ax.legend(); ax.grid(alpha=0.3)
ax.text(0.05, 0.90, f"Corr={corr:.3f}", transform=ax.transAxes, fontsize=11)

plt.tight_layout()
out_path = "neural_sde_v2_results.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved → {out_path}")
