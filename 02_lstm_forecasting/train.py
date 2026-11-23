#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM training script — NASA POWER 44-year data
Multi-variable LSTM, all outputs saved for paper figures + frost analysis
"""

import os, sys, json, time, argparse, logging, warnings
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore")

PROJECT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODELS_DIR = PROJECT_DIR / "models"
LOGS_DIR = PROJECT_DIR / "logs"
TBOARD_DIR = PROJECT_DIR / "tensorboard_logs"
for d in [MODELS_DIR, LOGS_DIR, TBOARD_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_file = LOGS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()])
log = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, output_steps, dropout=0.3):
        super().__init__()
        self.output_steps = output_steps
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size * output_steps),
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out.view(-1, self.output_steps, self.output_size)


def load_and_prepare(args):
    """Load data, normalize, create sequences."""
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        log.error("Processed data not found. Run download_data.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path, parse_dates=["Date"])
    val_df = pd.read_csv(PROCESSED_DIR / "val.csv", parse_dates=["Date"])
    test_df = pd.read_csv(PROCESSED_DIR / "test.csv", parse_dates=["Date"])
    feature_cols = [c for c in train_df.columns if c != "Date"]

    log.info(f"Data: train={len(train_df)}d, val={len(val_df)}d, test={len(test_df)}d")
    log.info(f"Variables: {feature_cols}")

    # Z-score normalization on training set stats
    train_mean = train_df[feature_cols].mean().values.astype(np.float32)
    train_std = train_df[feature_cols].std().values.astype(np.float32)
    train_std[train_std < 1e-6] = 1.0

    def norm(df): return (df[feature_cols].values.astype(np.float32) - train_mean) / train_std

    norm_train = norm(train_df)
    norm_val = norm(val_df)
    norm_test = norm(test_df)

    # Save normalization params for visualize.py
    np.savez(str(MODELS_DIR / "norm_params.npz"), mean=train_mean, std=train_std,
             cols=feature_cols)

    def make_seq(data, dates, si, so):
        X, Y, T = [], [], []
        for i in range(len(data) - si - so + 1):
            X.append(data[i:i+si])
            Y.append(data[i+si:i+si+so])
            T.append(dates[i+si])
        return np.array(X), np.array(Y), np.array(T)

    Xtr, Ytr, _ = make_seq(norm_train, train_df["Date"].values, args.input_steps, args.output_steps)
    Xva, Yva, _ = make_seq(norm_val, val_df["Date"].values, args.input_steps, args.output_steps)
    Xte, Yte, Tte = make_seq(norm_test, test_df["Date"].values, args.input_steps, args.output_steps)

    log.info(f"Sequences ({args.input_steps}d -> {args.output_steps}d): "
             f"train={Xtr.shape}, val={Xva.shape}, test={Xte.shape}")

    def make_loader(X, Y, bs, shuffle):
        ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
        return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True)

    train_loader = make_loader(Xtr, Ytr, args.batch_size, True)
    val_loader = make_loader(Xva, Yva, args.batch_size, False)
    test_loader = make_loader(Xte, Yte, args.batch_size, False)

    data_info = {
        "feature_cols": feature_cols, "input_size": len(feature_cols),
        "output_size": len(feature_cols), "input_steps": args.input_steps,
        "output_steps": args.output_steps,
        "norm_mean": train_mean, "norm_std": train_std,
        "X_test": Xte, "Y_test": Yte, "T_test": Tte,
    }
    return train_loader, val_loader, test_loader, data_info


def train_model(args, train_loader, val_loader, data_info):
    input_size = data_info["input_size"]
    output_size = data_info["output_size"]

    model = LSTMPredictor(input_size, args.hidden_size, args.num_layers,
                          output_size, args.output_steps, args.dropout).to(args.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)
    tb_writer = SummaryWriter(log_dir=str(TBOARD_DIR))

    log.info(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    log.info(f"Training: epochs={args.epochs}, batch={args.batch_size}, lr={args.lr}")

    best_val = float("inf")
    early_stop = 0
    history = {"epoch": [], "train_loss": [], "val_loss": []}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(args.device), by.to(args.device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * bx.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(args.device), by.to(args.device)
                loss = criterion(model(bx), by)
                val_loss += loss.item() * bx.size(0)
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)

        if epoch == 1 or epoch % 10 == 0:
            log.info(f"Epoch {epoch:3d} | train={train_loss:.6f} val={val_loss:.6f} lr={current_lr:.2e}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "best_model.pt")
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= args.patience:
                log.info(f"Early stopping at epoch {epoch}")
                break

        if epoch % args.save_period == 0:
            torch.save(model.state_dict(), MODELS_DIR / f"checkpoint_epoch_{epoch}.pt")

    torch.save(model.state_dict(), MODELS_DIR / "last_model.pt")
    pd.DataFrame(history).to_csv(MODELS_DIR / "training_history.csv", index=False)
    tb_writer.close()

    elapsed = time.time() - start_time
    log.info(f"Training complete: {int(elapsed//3600)}h{int((elapsed%3600)//60)}m")
    log.info(f"Best val loss: {best_val:.6f}")
    model.load_state_dict(torch.load(MODELS_DIR / "best_model.pt"))
    return model


def test_and_save(model, test_loader, data_info, args):
    """Test and save predictions + frost climatology stats."""
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(args.device)
            all_preds.append(model(bx).cpu().numpy())
            all_targets.append(by.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Denormalize
    mean = data_info["norm_mean"]
    std = data_info["norm_std"]
    preds_orig = all_preds * std + mean
    targets_orig = all_targets * std + mean

    # Temperature metrics
    vi = list(data_info["feature_cols"]).index("T2M")
    errs = targets_orig[:, 0, vi] - preds_orig[:, 0, vi]
    mae = np.mean(np.abs(errs))
    rmse = np.sqrt(np.mean(errs**2))
    ssr = np.sum(errs**2)
    sst = np.sum((targets_orig[:, 0, vi] - np.mean(targets_orig[:, 0, vi]))**2)
    r2 = 1 - ssr / max(sst, 1e-10)
    log.info(f"\nTemperature (T2M) — MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")

    # Save predictions
    np.savez(MODELS_DIR / "predictions.npz",
             preds=preds_orig, targets=targets_orig,
             norm_mean=mean, norm_std=std,
             T_test=data_info["T_test"].astype(str),
             cols=data_info["feature_cols"])

    # Save frost climatology (Route C): for each calendar day, compute
    # P(WS < 3 m/s) and P(RH > 60%) from training data
    log.info("Computing frost climatology (Route C)...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv", parse_dates=["Date"])
    train_df["doy"] = train_df["Date"].dt.dayofyear

    frost_clim = []
    for doy in range(1, 367):
        day_data = train_df[train_df["doy"] == doy]
        if len(day_data) > 0:
            p_wind = np.mean(day_data["WS2M"].values < 3.0)
            p_hum = np.mean(day_data["RH2M"].values > 60.0)
            p_frost = np.mean(day_data["T2M"].values < 0.0)
        else:
            p_wind, p_hum, p_frost = 0, 0, 0
        frost_clim.append({"doy": doy, "P_wind_lt_3": p_wind, "P_hum_gt_60": p_hum, "P_frost": p_frost})

    pd.DataFrame(frost_clim).to_csv(MODELS_DIR / "frost_climatology.csv", index=False)
    log.info("Frost climatology saved")
    log.info(f"Daily P(WS<3) range: {min(d['P_wind_lt_3'] for d in frost_clim):.2f} - "
             f"{max(d['P_wind_lt_3'] for d in frost_clim):.2f}")
    log.info("Next: python visualize.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--input_steps", type=int, default=14)
    parser.add_argument("--output_steps", type=int, default=7)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--save_period", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)

    log.info(f"Device: {args.device}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, data_info = load_and_prepare(args)
    model = train_model(args, train_loader, val_loader, data_info)
    test_and_save(model, test_loader, data_info, args)
    log.info("All complete!")
