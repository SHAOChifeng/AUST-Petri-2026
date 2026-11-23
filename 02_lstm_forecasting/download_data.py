#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NASA POWER data download + preprocessing (44 years)
Covers 1981-01-01 to 2024-12-31 for Hefei (31.8N, 117.2E)
"""

import os, sys, json, csv, logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import numpy as np

log_dir = Path(__file__).parent / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).parent.resolve()
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

LAT, LON = 31.8, 117.2  # Hefei
PARAMETERS = "T2M,PRECTOTCORR,RH2M,WS2M,PS,ALLSKY_SFC_SW_DWN"
START = "19810101"
END = "20241231"


def download_nasa_power():
    """Download data from NASA POWER API."""
    url = (f"https://power.larc.nasa.gov/api/temporal/daily/point?"
           f"parameters={PARAMETERS}&community=AG&"
           f"longitude={LON}&latitude={LAT}&"
           f"start={START}&end={END}&format=JSON")

    csv_path = RAW_DIR / f"nasa_power_{START}_{END}.csv"
    if csv_path.exists():
        log.info(f"Local file exists: {csv_path}")
        return csv_path

    log.info(f"Downloading NASA POWER data: {LAT}N, {LON}E, {START}-{END} (44 years)")
    try:
        resp = requests.get(url, timeout=120)
        if resp.status_code != 200:
            log.warning(f"API returned {resp.status_code}")
            return None
        data = resp.json()
        params = data["properties"]["parameter"]
        dates = sorted(params["T2M"].keys())
        log.info(f"Got {len(dates)} days of data")

        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Date", "T2M", "PRECTOTCORR", "RH2M", "WS2M", "PS", "ALLSKY_SFC_SW_DWN"])
            for d in dates:
                w.writerow([d, params["T2M"].get(d, ""), params["PRECTOTCORR"].get(d, ""),
                            params["RH2M"].get(d, ""), params["WS2M"].get(d, ""),
                            params["PS"].get(d, ""), params["ALLSKY_SFC_SW_DWN"].get(d, "")])
        log.info(f"Saved: {csv_path}")
        return csv_path
    except Exception as e:
        log.warning(f"Download failed: {e}")
        return None


def preprocess(csv_path):
    """Preprocess: clean, normalize, create splits."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
    df = df.sort_values("Date").reset_index(drop=True)

    for col in df.columns:
        if col != "Date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.interpolate(method="linear", limit_direction="both").ffill().bfill()

    feature_cols = [c for c in df.columns if c != "Date"]
    log.info(f"Data shape: {len(df)} days, {len(feature_cols)} variables")
    for col in feature_cols:
        log.info(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")

    # Save full processed data
    processed_path = PROCESSED_DIR / "nasa_power_processed.csv"
    df.to_csv(processed_path, index=False)

    # Save normalization params
    norm_params = {col: {"mean": float(df[col].mean()), "std": float(df[col].std())}
                   for col in feature_cols}
    with open(str(PROCESSED_DIR / "norm_params.json"), "w") as f:
        json.dump(norm_params, f, indent=2)

    # Chronological split: 80% train, 10% val, 10% test
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    splits = {
        "train": df.iloc[:train_end],
        "val": df.iloc[train_end:val_end],
        "test": df.iloc[val_end:],
    }
    for name, sdf in splits.items():
        sdf.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        log.info(f"  {name}: {len(sdf)} days ({sdf['Date'].iloc[0].date()} to {sdf['Date'].iloc[-1].date()})")

    return df


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("NASA POWER data download (44 years: 1981-2024)")
    log.info("=" * 60)
    csv_path = download_nasa_power()
    if csv_path is None:
        log.error("Download failed. Place CSV in data/raw/ manually.")
        sys.exit(1)
    preprocess(csv_path)
    log.info("Next: python train.py")
