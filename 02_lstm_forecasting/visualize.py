#!/usr/bin/env python3
"""LSTM visualization v2 — Fixed: full temp range including negatives, MDPI-compliant."""
import os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "figure.dpi": 200, "savefig.dpi": 200, "savefig.bbox": "tight",
})
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore")

NATURE = {"red":"#D62728","blue":"#2B6FAF","green":"#2C8350","orange":"#E88326","purple":"#8B5F9E","teal":"#1F9B8E","grey":"#7F7F7F"}
COLORS = [NATURE["blue"],NATURE["red"],NATURE["green"],NATURE["orange"],NATURE["purple"],NATURE["teal"]]

# Auto-detect project root (works on Windows and Linux)
BASE = Path(__file__).parent.resolve()
MDIR = BASE / "models"; OUTD = BASE / "output"; FD = OUTD / "figures"; ED = OUTD / "extra"
for d in [FD, ED]: d.mkdir(parents=True, exist_ok=True)

data = np.load(MDIR / "predictions.npz", allow_pickle=True)
preds = data["preds"]; targets = data["targets"]
T_test = data["T_test"].astype(str); cols = list(data["cols"])
hist = pd.read_csv(MDIR / "training_history.csv")
frost_clim = pd.read_csv(MDIR / "frost_climatology.csv")
vi_t = cols.index("T2M"); vi_w = cols.index("WS2M"); vi_h = cols.index("RH2M")

def sv(fig, name):
    for d in [FD, ED]: fig.savefig(str(d/name), dpi=200, bbox_inches="tight")
    print(f"  Saved: {name}")

# ── FIGURE 15: Temperature prediction (with full negative range) ──
def make_fig15():
    n_show = min(180, len(preds))
    start = max(0, len(preds) - n_show)
    times = pd.to_datetime(T_test[start:start + n_show])
    fig, ax = plt.subplots(figsize=(7, 3.2))

    actual = targets[start:start + n_show, 0, vi_t]
    predicted = preds[start:start + n_show, 0, vi_t]

    ax.plot(times, actual, color="#222222", lw=0.8, alpha=0.5, label="Ground truth")
    ax.plot(times, predicted, color=NATURE["red"], lw=1.0, label="LSTM prediction")

    err = np.abs(predicted - actual)
    sd = np.std(err)
    ax.fill_between(times, predicted - 1.96*sd, predicted + 1.96*sd,
                    color=NATURE["red"], alpha=0.08, label="95% CI")

    mae = np.mean(err)
    r2 = 1 - np.sum((predicted-actual)**2) / max(np.sum((actual-actual.mean())**2), 1e-10)

    ax.text(0.98, 0.05, f"MAE={mae:.2f}°C  R\u00b2={r2:.3f}", transform=ax.transAxes,
            fontsize=7, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", lw=0.5))

    # Explicit y-limit to show full range including negatives
    ymin = min(actual.min(), predicted.min()) - 2
    ymax = max(actual.max(), predicted.max()) + 2
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel("Date"); ax.set_ylabel("Temperature (\u00b0C)")
    ax.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=6.5, ncol=2)
    ax.tick_params(direction="in", length=3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.axhline(y=0, color="#888888", ls=":", lw=0.5, alpha=0.5)
    plt.tight_layout(); sv(fig, "fig15_lstm_prediction.png"); plt.close()
    print(f"  T2M: MAE={mae:.2f}, R2={r2:.3f}, temp range=({ymin:.1f}, {ymax:.1f})")

# ── FIGURE S1: Loss ──
def make_S1():
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    ax.plot(hist["epoch"], hist["train_loss"], color=NATURE["blue"], lw=0.8, label="Training")
    ax.plot(hist["epoch"], hist["val_loss"], color=NATURE["red"], lw=0.8, label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
    ax.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax.tick_params(direction="in", length=3)
    min_ep = hist.loc[hist["val_loss"].idxmin(), "epoch"]
    ax.axvline(min_ep, color="#888888", ls=":", lw=0.6)
    ylo, yhi = ax.get_ylim()
    ax.annotate(f"Best: epoch {int(min_ep)}", xy=(min_ep, hist["val_loss"].min()),
                xytext=(min_ep + max(2, len(hist)*0.03), yhi*0.85),
                arrowprops=dict(arrowstyle="->", color="#888888", lw=0.7), fontsize=7)
    plt.tight_layout(); sv(fig, "S1_loss_curve.png"); plt.close()

# ── FIGURE S2: Six variables ──
def make_S2():
    n_show = min(100, len(preds)); start = max(0, len(preds)-n_show)
    times = pd.to_datetime(T_test[start:start+n_show])
    fig, axes = plt.subplots(3, 2, figsize=(7, 5.5)); axs = axes.flatten()
    names = {"T2M":"Temperature","PRECTOTCORR":"Precipitation","RH2M":"Humidity",
             "WS2M":"Wind speed","PS":"Pressure","ALLSKY_SFC_SW_DWN":"Solar radiation"}
    for vi in range(min(len(cols),6)):
        ax = axs[vi]
        ax.plot(times, targets[start:start+n_show,0,vi], color="#222222", lw=0.7, alpha=0.5, label="Actual")
        ax.plot(times, preds[start:start+n_show,0,vi], color=NATURE["red"], lw=0.8, ls="--", label="Pred")
        ax.set_ylabel(names.get(cols[vi], cols[vi]), fontsize=7)
        ax.legend(fontsize=6, frameon=True, fancybox=False, edgecolor="#CCCCCC")
        ax.tick_params(direction="in", length=2)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.tight_layout(); sv(fig, "S2_all_variables.png"); plt.close()

# ── FIGURE S3: Frost climatology (emphasised P_frost line) ──
def make_S3():
    fig, ax = plt.subplots(figsize=(7, 3.2))

    ax.plot(frost_clim["doy"], frost_clim["P_wind_lt_3"], color=NATURE["blue"], lw=0.8, label="P(WS < 3 m/s)")
    ax.plot(frost_clim["doy"], frost_clim["P_hum_gt_60"], color=NATURE["green"], lw=0.8, label="P(RH > 60%)")

    # P_frost with bold line and visible fill to highlight winter peak
    ax.plot(frost_clim["doy"], frost_clim["P_frost"], color=NATURE["red"], lw=2.0, label="P(T < 0 degC)")
    ax.fill_between(frost_clim["doy"], 0, frost_clim["P_frost"], color=NATURE["red"], alpha=0.15)

    # Mark the winter peak value
    max_idx = frost_clim["P_frost"].idxmax()
    max_doy = frost_clim.loc[max_idx, "doy"]
    max_val = frost_clim.loc[max_idx, "P_frost"]
    ax.annotate(f"Peak: {max_val:.0%}", xy=(max_doy, max_val),
                xytext=(max_doy + 30, max_val + 0.12),
                arrowprops=dict(arrowstyle="->", color=NATURE["red"], lw=0.8),
                fontsize=7, color=NATURE["red"], fontweight="bold")

    ax.axhline(y=0.7, color="#888888", ls="--", lw=0.6, label="Threshold (0.7)")
    ax.set_xlabel("Day of year"); ax.set_ylabel("Probability")
    ax.legend(frameon=True, fancybox=False, edgecolor="#CCCCCC", fontsize=6.5, ncol=2)
    ax.tick_params(direction="in", length=3)
    ax.set_xlim(0, 366); ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335])
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    plt.tight_layout(); sv(fig, "S3_frost_climatology.png"); plt.close()
    print(f"  P_frost max={max_val:.3f} on day {max_doy}, nonzero days={(frost_clim['P_frost']>0).sum()}")

# ── FIGURE S4: Frost prediction ──
def make_S4():
    n_test = len(preds); test_dates = pd.to_datetime(T_test)
    test_doy = np.array([d.dayofyear for d in test_dates])
    p_wind = np.zeros(n_test); p_hum = np.zeros(n_test)
    for i,doy in enumerate(test_doy):
        m = frost_clim["doy"] == doy
        if m.any():
            p_wind[i] = frost_clim.loc[m, "P_wind_lt_3"].values[0]
            p_hum[i] = frost_clim.loc[m, "P_hum_gt_60"].values[0]
    
    temp_pred = preds[:,0,vi_t]; temp_actual = targets[:,0,vi_t]
    frost_pred = (temp_pred < 0) & (p_wind > 0.7) & (p_hum > 0.7)
    frost_actual = (temp_actual < 0) & (targets[:,0,vi_w] < 3.0) & (targets[:,0,vi_h] > 60)
    
    tp = np.sum(frost_actual & frost_pred); fn = np.sum(frost_actual & ~frost_pred)
    fp = np.sum(~frost_actual & frost_pred); tn = n_test - tp - fn - fp
    recall = tp / max(tp+fn, 1); prec = tp / max(tp+fp, 1); f1 = 2*tp/max(2*tp+fp+fn, 1)
    far = fp / max(fp+tn, 1)

    fig, axes = plt.subplots(1, 3, figsize=(8, 3.2))

    # Panel A: Confusion matrix
    ax = axes[0]; cm = np.array([[tp, fn], [fp, tn]])
    ax.matshow([[1,0],[0,1]], cmap="Blues", alpha=0.1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["Frost","No frost"]); ax.set_yticklabels(["Pred frost","Pred no frost"], fontsize=6)
    ax.set_title("Confusion matrix", fontsize=7, fontweight="bold")

    # Panel B: Metrics
    ax = axes[1]; metrics = {"Hit rate": recall, "Precision": prec, "F1": f1, "False alarm\nrate": far}
    clrs = [NATURE["green"], NATURE["blue"], NATURE["orange"], NATURE["red"]]
    ax.bar(range(len(metrics)), list(metrics.values()), color=clrs, width=0.55, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(list(metrics.keys()), fontsize=6)
    ax.set_ylim(0, max(1.0, max(metrics.values())*1.2)); ax.set_ylabel("Score")
    ax.tick_params(direction="in", length=2)
    ax.set_title("Performance metrics", fontsize=7, fontweight="bold")
    for i,(k,v) in enumerate(metrics.items()):
        ax.text(i, v+0.02, f"{v:.2f}", ha="center", fontsize=6.5, fontweight="bold")

    # Panel C: Timeline
    ax = axes[2]; n_show = min(200, n_test); start = max(0, n_test-n_show)
    times = test_dates[start:start+n_show]
    ax.fill_between(times, 0, 1, where=frost_actual[start:start+n_show], color=NATURE["red"], alpha=0.25, lw=0)
    ax.fill_between(times, 0, 1, where=frost_pred[start:start+n_show], color=NATURE["orange"], alpha=0.4, lw=0)
    ax.set_ylim(0, 1.1); ax.set_xlabel("Date")
    ax.set_title("Frost timeline", fontsize=7, fontweight="bold")
    ax.legend(["Actual","Predicted"], fontsize=5.5, frameon=True, fancybox=False, edgecolor="#CCCCCC")
    ax.tick_params(direction="in", length=2); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout(); sv(fig, "S4_frost_prediction.png"); plt.close()
    print(f"  Frost: hit={recall:.1%}, prec={prec:.1%}, F1={f1:.3f}")

# ===================== 自动发布到论文目录（带版本号） =====================

PAPER_FIG_DIR = BASE.parent / "figures"

LSTM_FIG_MAP = [
    (FD / "fig15_lstm_prediction.png", "fig15_lstm_prediction"),
    (FD / "S1_loss_curve.png", "S1_loss_curve"),
    (FD / "S2_all_variables.png", "S2_all_variables"),
    (FD / "S3_frost_climatology.png", "S3_frost_climatology"),
    (FD / "S4_frost_prediction.png", "S4_frost_prediction"),
]


def publish_figure(src_path, paper_name):
    """复制到论文目录，hash 不变则跳过，变化则归档旧版为 _v{N}"""
    if not src_path.exists():
        print(f"  WARNING: {paper_name}.png source not found: {src_path}")
        return False

    import hashlib, shutil
    paper_dir = PAPER_FIG_DIR
    paper_dir.mkdir(parents=True, exist_ok=True)

    base_file = paper_dir / f"{paper_name}.png"
    new_hash = hashlib.md5(src_path.read_bytes()).hexdigest()

    if base_file.exists():
        old_hash = hashlib.md5(base_file.read_bytes()).hexdigest()
        if old_hash == new_hash:
            print(f"  {paper_name}.png: unchanged (skipped)")
            return True
        v = 1
        while (paper_dir / f"{paper_name}_v{v}.png").exists():
            v += 1
        shutil.copy(str(base_file), str(paper_dir / f"{paper_name}_v{v}.png"))
        print(f"  {paper_name}.png: archived as _v{v}.png")

    shutil.copy(str(src_path), str(base_file))
    print(f"  {paper_name}.png: updated ✓")
    return True


def publish_lstm_figures():
    print("\n" + "=" * 60)
    print("Publishing LSTM figures to paper directory...")
    print("=" * 60)
    for src, name in LSTM_FIG_MAP:
        publish_figure(src, name)
    print("LSTM publishing done.")


if __name__ == "__main__":
    print("Generating figures (fixed: full temp range, visible frost line)...")
    make_fig15(); make_S1(); make_S2(); make_S3(); make_S4()
    print(f"\nDone. Figures in: {FD}")
    publish_lstm_figures()
