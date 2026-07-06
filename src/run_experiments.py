from __future__ import annotations

import argparse
import math
import os
import random
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

# Anaconda, PyTorch and plotting libraries may load different OpenMP DLLs on
# Windows. Set this before importing torch/numpy/matplotlib so the script can
# finish and print the result table in the common local course environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)

SUM_COLUMNS = [
    "global_active_power",
    "global_reactive_power",
    "sub_metering_1",
    "sub_metering_2",
    "sub_metering_3",
]
MEAN_COLUMNS = ["voltage", "global_intensity"]
TARGET = "global_active_power"


def log(message: str = "") -> None:
    """Print immediately so long training runs still show screenshot-friendly progress."""

    print(message, flush=True)


@dataclass(frozen=True)
class Dataset:
    """Daily train/test split plus the feature list used by all models."""

    train: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    target: str = TARGET


def set_seed(seed: int) -> None:
    """Make one experiment run reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column-name variants in the UCI raw file."""

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df.rename(
        columns={
            "sub_metering3": "sub_metering_3",
            "globalactivepower": "global_active_power",
            "globalreactivepower": "global_reactive_power",
            "globalintensity": "global_intensity",
        }
    )


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Build a single datetime column from date/time fields."""

    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns and "time" in df.columns:
        dt = df["date"].astype(str) + " " + df["time"].astype(str)
        df["datetime"] = pd.to_datetime(dt, errors="coerce", dayfirst=True)
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    else:
        raise ValueError("Data must contain either datetime or date/time columns.")
    return df.dropna(subset=["datetime"])


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking columns and treat UCI '?' markers as missing values."""

    df = df.copy()
    for col in df.columns:
        if col not in {"date", "time", "datetime"}:
            df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")
    return df


def daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level records to daily samples required by the assignment."""

    df = parse_datetime(to_numeric(normalize_columns(df)))
    df["day"] = df["datetime"].dt.floor("D")
    agg: dict[str, str] = {}
    for col in SUM_COLUMNS:
        if col in df.columns:
            agg[col] = "sum"
    for col in MEAN_COLUMNS:
        if col in df.columns:
            agg[col] = "mean"
    if TARGET not in agg:
        raise ValueError(f"Required target column {TARGET!r} is missing.")

    # Course rules: energy-like columns are summed by day, voltage/current are averaged.
    daily = df.groupby("day", as_index=False).agg(agg).sort_values("day")
    daily = daily.set_index("day").asfreq("D").reset_index()
    value_cols = [c for c in daily.columns if c != "day"]
    # Calendar reindexing plus interpolation ensures every window is consecutive in time.
    daily[value_cols] = daily[value_cols].interpolate(limit_direction="both")
    daily[value_cols] = daily[value_cols].fillna(daily[value_cols].median(numeric_only=True))

    if {"sub_metering_1", "sub_metering_2", "sub_metering_3"}.issubset(daily.columns):
        daily["sub_metering_remainder"] = (
            daily[TARGET] * 1000.0 / 60.0
            - daily["sub_metering_1"]
            - daily["sub_metering_2"]
            - daily["sub_metering_3"]
        )
    return daily


def download_uci() -> Path:
    """Download and extract the public UCI dataset when it is not already local."""

    DATA_DIR.mkdir(exist_ok=True)
    txt_path = DATA_DIR / "household_power_consumption.txt"
    zip_path = DATA_DIR / "household_power_consumption.zip"
    if txt_path.exists():
        return txt_path
    log("Downloading UCI Individual household electric power consumption dataset...")
    urllib.request.urlretrieve(UCI_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("household_power_consumption.txt", DATA_DIR)
    return txt_path


def load_dataset() -> Dataset:
    """Load UCI raw minute-level data and create a time-ordered train/test split."""

    raw_path = download_uci()
    raw = pd.read_csv(raw_path, sep=";", low_memory=False)
    daily = daily_aggregate(raw)
    # Hold out the last quarter, with at least 365 days reserved for long-horizon testing.
    split = min(int(len(daily) * 0.75), len(daily) - 365)
    if split < 455:
        raise ValueError("UCI data is too short for 90-day input and 365-day output.")
    train, test = daily.iloc[:split].copy(), daily.iloc[split:].copy()
    features = [c for c in train.columns if c != "day"]
    return Dataset(train=train, test=test, features=features)


def standardize(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str):
    """Standardize with training statistics only to avoid test leakage."""

    x_mean = train[features].mean()
    x_std = train[features].std().replace(0, 1.0)
    y_mean = float(train[target].mean())
    y_std = float(train[target].std() or 1.0)
    train_x = ((train[features] - x_mean) / x_std).to_numpy(dtype=np.float32)
    test_x = ((test[features] - x_mean) / x_std).to_numpy(dtype=np.float32)
    train_y = ((train[target] - y_mean) / y_std).to_numpy(dtype=np.float32)
    test_y = ((test[target] - y_mean) / y_std).to_numpy(dtype=np.float32)
    return train_x, test_x, train_y, test_y, y_mean, y_std


def make_windows(x: np.ndarray, y: np.ndarray, input_len: int, horizon: int):
    """Create samples shaped as past input_len days -> future horizon days."""

    xs, ys = [], []
    end = len(x) - input_len - horizon + 1
    for i in range(max(0, end)):
        xs.append(x[i : i + input_len])
        ys.append(y[i + input_len : i + input_len + horizon])
    if not xs:
        raise ValueError(
            f"Not enough daily rows for input={input_len}, horizon={horizon}. "
            f"Need at least {input_len + horizon}, got {len(x)}."
        )
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


class LSTMForecaster(nn.Module):
    """Fully trainable LSTM encoder with a multi-step regression head."""

    name = "LSTM"

    def __init__(self, n_features: int, hidden: int, layers: int, horizon: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class PositionalEncoding(nn.Module):
    """Sinusoidal position encoding for daily time steps."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """Fully trainable Transformer encoder for multi-step forecasting."""

    name = "Transformer"

    def __init__(
        self,
        n_features: int,
        hidden: int,
        layers: int,
        heads: int,
        horizon: int,
        dropout: float,
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden)
        self.pos = PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.input_proj(x))
        z = self.encoder(z)
        return self.head(z[:, -1])


class CNNTransformerForecaster(nn.Module):
    """Proposed model: local temporal convolution followed by Transformer encoding."""

    name = "CNN-Transformer"

    def __init__(
        self,
        n_features: int,
        hidden: int,
        layers: int,
        heads: int,
        horizon: int,
        dropout: float,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pos = PositionalEncoding(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv1d expects [batch, channels, time], then Transformer expects [batch, time, hidden].
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        z = self.encoder(self.pos(z))
        return self.head(z[:, -1])


def build_model(name: str, n_features: int, horizon: int, args) -> nn.Module:
    if name == "LSTM":
        return LSTMForecaster(n_features, args.hidden, args.layers, horizon, args.dropout)
    if name == "Transformer":
        return TransformerForecaster(n_features, args.hidden, args.layers, args.heads, horizon, args.dropout)
    if name == "CNN-Transformer":
        return CNNTransformerForecaster(n_features, args.hidden, args.layers, args.heads, horizon, args.dropout)
    raise ValueError(f"Unknown model: {name}")


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_model(model: nn.Module, xtr: np.ndarray, ytr: np.ndarray, args, device: torch.device) -> nn.Module:
    """Train all model parameters end to end with backpropagation."""

    n = len(xtr)
    val_n = max(1, int(n * args.val_ratio))
    tr_x, val_x = xtr[:-val_n], xtr[-val_n:]
    tr_y, val_y = ytr[:-val_n], ytr[-val_n:]
    train_loader = make_loader(tr_x, tr_y, args.batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, args.batch_size, shuffle=False)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, loss_fn, device)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device) -> float:
    model.eval()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        losses.append(float(loss_fn(model(xb), yb).cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    preds = []
    loader = make_loader(x, np.zeros((len(x), 1), dtype=np.float32), batch_size, shuffle=False)
    for xb, _ in loader:
        preds.append(model(xb.to(device)).cpu().numpy())
    return np.vstack(preds)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    return float(np.mean(err * err)), float(np.mean(np.abs(err)))


def draw_table(summary: pd.DataFrame, path: Path):
    """Render the mean/std metrics as a PNG table for direct inclusion in the report."""

    fig, ax = plt.subplots(figsize=(10, max(2.5, 0.45 * (len(summary) + 1))))
    ax.axis("off")
    display = summary.copy()
    for col in ["mse_mean", "mse_std", "mae_mean", "mae_std"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_curve(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path):
    """Render prediction vs. ground-truth curves."""

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(y_true, label="Ground Truth", linewidth=1.7)
    ax.plot(y_pred, label="Prediction", linewidth=1.7)
    ax.set_title(title)
    ax.set_xlabel("Day index")
    ax.set_ylabel("Global active power")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(args):
    OUTPUT_DIR.mkdir(exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device in {"cpu", "cuda"}:
        device = torch.device(args.device)
    log("=" * 72)
    log("Household power forecasting experiment")
    log("=" * 72)
    log(f"Using device: {device}")
    log(f"Python executable: {sys.executable}")
    log(
        "Config: "
        f"runs={args.runs}, epochs={args.epochs}, input_len={args.input_len}, "
        f"hidden={args.hidden}, layers={args.layers}, batch_size={args.batch_size}"
    )

    ds = load_dataset()
    log(f"Daily samples: train={len(ds.train)}, test={len(ds.test)}, features={len(ds.features)}")
    train_x, test_x, train_y, test_y, y_mean, y_std = standardize(
        ds.train, ds.test, ds.features, ds.target
    )
    model_names = ["LSTM", "Transformer", "CNN-Transformer"]
    horizons = [90, 365]
    rows = []
    prediction_figures = {}

    for horizon in horizons:
        log("")
        log(f"Preparing horizon={horizon} task")
        # Short-term and long-term tasks are trained independently as required.
        xtr, ytr = make_windows(train_x, train_y, args.input_len, horizon)
        xte, yte = make_windows(
            # Prepend the last training days so test windows have full historical context.
            np.vstack([train_x[-args.input_len :], test_x]),
            np.r_[train_y[-args.input_len :], test_y],
            args.input_len,
            horizon,
        )
        if args.max_train_samples:
            xtr = xtr[-args.max_train_samples :]
            ytr = ytr[-args.max_train_samples :]
        log(f"Window samples: train={len(xtr)}, test={len(xte)}")

        for model_name in model_names:
            for run_id in range(args.runs):
                seed = args.seed + run_id
                set_seed(seed)
                log(f"Training {model_name}, horizon={horizon}, run={run_id + 1}/{args.runs}...")
                model = build_model(model_name, xtr.shape[-1], horizon, args)
                model = train_model(model, xtr, ytr, args, device)
                pred = predict(model, xte, args.batch_size, device)
                true_real = yte * y_std + y_mean
                pred_real = pred * y_std + y_mean
                mse, mae = metrics(true_real, pred_real)
                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "run": run_id + 1,
                        "mse": mse,
                        "mae": mae,
                    }
                )
                # Use the last test forecast from the first run for report curves.
                if run_id == 0:
                    prediction_figures[(model_name, horizon)] = (true_real[-1], pred_real[-1])
                log(
                    f"{model_name:16s} horizon={horizon:3d} "
                    f"run={run_id + 1} mse={mse:.3f} mae={mae:.3f}"
                )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        metrics_df.groupby(["model", "horizon"])
        .agg(
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
        )
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    display = summary.copy()
    for col in ["mse_mean", "mse_std", "mae_mean", "mae_std"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    log("")
    log("=" * 72)
    log(f"Final {args.runs}-run summary (mean +/- std)")
    log("=" * 72)
    log(display.to_string(index=False))

    draw_table(summary, OUTPUT_DIR / "metrics_summary.png")

    for (model_name, horizon), (true, pred) in prediction_figures.items():
        safe_model = model_name.lower().replace("-", "_")
        draw_curve(
            true,
            pred,
            f"{model_name} forecast, horizon={horizon}",
            OUTPUT_DIR / f"curve_{safe_model}_{horizon}.png",
        )
    log("")
    log("Generated report artifacts:")
    for path in [
        OUTPUT_DIR / "metrics.csv",
        OUTPUT_DIR / "metrics_summary.csv",
        OUTPUT_DIR / "metrics_summary.png",
    ]:
        log(f"- {path}")
    log(f"Saved outputs to {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
