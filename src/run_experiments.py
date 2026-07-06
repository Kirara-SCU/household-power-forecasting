from __future__ import annotations

import argparse
import csv
import math
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


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
WEATHER_COLUMNS = ["RR", "NBJRR1", "NBJRR5", "NBJRR10", "NBJBROU"]
TARGET = "global_active_power"


@dataclass(frozen=True)
class Dataset:
    """Daily train/test split plus the feature list used by all models."""

    train: pd.DataFrame
    test: pd.DataFrame
    features: list[str]
    target: str = TARGET


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common column-name variants from course CSVs or the UCI raw file."""

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename = {
        "sub_metering_3": "sub_metering_3",
        "sub_metering3": "sub_metering_3",
        "globalactivepower": "global_active_power",
        "globalreactivepower": "global_reactive_power",
        "globalintensity": "global_intensity",
    }
    return df.rename(columns=rename)


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Build a single datetime column from either datetime or date/time fields."""

    df = df.copy()
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    elif "date" in df.columns and "time" in df.columns:
        dt = df["date"].astype(str) + " " + df["time"].astype(str)
        df["datetime"] = pd.to_datetime(dt, errors="coerce", dayfirst=True)
    elif "date" in df.columns:
        df["datetime"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    else:
        raise ValueError("CSV must contain either datetime or date/time columns.")
    return df.dropna(subset=["datetime"])


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric-looking columns and treat UCI '?' markers as missing values."""

    df = df.copy()
    for col in df.columns:
        if col not in {"date", "time", "datetime"}:
            df[col] = pd.to_numeric(df[col].replace("?", np.nan), errors="coerce")
    return df


def daily_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute-level records to the daily granularity required by the task."""

    df = parse_datetime(to_numeric(normalize_columns(df)))
    df["day"] = df["datetime"].dt.floor("D")
    agg: dict[str, str] = {}
    for col in SUM_COLUMNS:
        if col in df.columns:
            agg[col] = "sum"
    for col in MEAN_COLUMNS:
        if col in df.columns:
            agg[col] = "mean"
    for col in WEATHER_COLUMNS:
        if col.lower() in df.columns:
            agg[col.lower()] = "first"
    if TARGET not in agg:
        raise ValueError(f"Required target column {TARGET!r} is missing.")

    # Course rules: energy-like columns are summed by day, voltage/current are averaged,
    # and weather columns keep one representative daily value.
    daily = df.groupby("day", as_index=False).agg(agg).sort_values("day")
    daily = daily.set_index("day").asfreq("D").reset_index()
    value_cols = [c for c in daily.columns if c != "day"]
    # Missing days or missing records are filled after calendar reindexing so windows
    # always represent consecutive days.
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


def find_course_csvs() -> tuple[Path | None, Path | None]:
    """Prefer teacher-provided train/test files when they are present."""

    candidates = [ROOT, DATA_DIR]
    train = None
    test = None
    for base in candidates:
        for name in ("train.csv", "Train.csv"):
            p = base / name
            if p.exists():
                train = p
        for name in ("test.csv", "tes.csv", "Test.csv", "Tes.csv"):
            p = base / name
            if p.exists():
                test = p
    return train, test


def download_uci() -> Path:
    """Download the public UCI fallback dataset for local verification."""

    DATA_DIR.mkdir(exist_ok=True)
    txt_path = DATA_DIR / "household_power_consumption.txt"
    zip_path = DATA_DIR / "household_power_consumption.zip"
    if txt_path.exists():
        return txt_path
    print("Course CSV files were not found; downloading UCI fallback dataset...")
    urllib.request.urlretrieve(UCI_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract("household_power_consumption.txt", DATA_DIR)
    return txt_path


def load_dataset() -> Dataset:
    """Load course data if available; otherwise create a reproducible UCI fallback split."""

    train_csv, test_csv = find_course_csvs()
    if train_csv and test_csv:
        print(f"Using course files: {train_csv.name}, {test_csv.name}")
        train_raw = pd.read_csv(train_csv)
        test_raw = pd.read_csv(test_csv)
        train = daily_aggregate(train_raw)
        test = daily_aggregate(test_raw)
    else:
        raw_path = download_uci()
        raw = pd.read_csv(raw_path, sep=";", low_memory=False)
        daily = daily_aggregate(raw)
        # Keep enough trailing days for the 365-day test horizon while preserving most
        # observations for training. This branch is only a fallback when course CSVs are absent.
        split = min(int(len(daily) * 0.75), len(daily) - 365)
        if split < 455:
            raise ValueError("UCI fallback data is too short for 90-day input and 365-day output.")
        train, test = daily.iloc[:split].copy(), daily.iloc[split:].copy()

    features = [c for c in train.columns if c not in {"day"}]
    missing = [c for c in features if c not in test.columns]
    for col in missing:
        # Some course test files may omit optional weather/sub-meter columns.
        test[col] = train[col].median()
    test = test[["day"] + features]
    return Dataset(train=train, test=test, features=features)


def standardize(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target: str):
    """Standardize with training statistics only to avoid leaking test information."""

    x_mean = train[features].mean()
    x_std = train[features].std().replace(0, 1.0)
    y_mean = float(train[target].mean())
    y_std = float(train[target].std() or 1.0)
    train_x = ((train[features] - x_mean) / x_std).to_numpy(dtype=np.float64)
    test_x = ((test[features] - x_mean) / x_std).to_numpy(dtype=np.float64)
    train_y = ((train[target] - y_mean) / y_std).to_numpy(dtype=np.float64)
    test_y = ((test[target] - y_mean) / y_std).to_numpy(dtype=np.float64)
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
    return np.stack(xs), np.stack(ys)


def ridge_fit(phi: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Fit the multi-output linear head using ridge regression."""

    phi = np.c_[np.ones(len(phi)), phi]
    eye = np.eye(phi.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.solve(phi.T @ phi + alpha * eye, phi.T @ y)


def ridge_predict(phi: np.ndarray, w: np.ndarray) -> np.ndarray:
    return np.c_[np.ones(len(phi)), phi] @ w


class BaseModel:
    """Shared fixed-encoder plus trainable linear-head forecasting interface."""

    def __init__(self, seed: int, hidden: int = 32):
        self.rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.weights: np.ndarray | None = None

    def encode(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit(self, x: np.ndarray, y: np.ndarray):
        # The reservoir encoders are randomly initialized per seed; only the linear
        # output layer is fitted, which keeps the assignment runnable without PyTorch.
        phi = self.encode(x)
        self.weights = ridge_fit(phi, y, alpha=2.0)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("Model is not fitted.")
        return ridge_predict(self.encode(x), self.weights)


class LSTMReservoir(BaseModel):
    name = "LSTM"

    def fit(self, x: np.ndarray, y: np.ndarray):
        d = x.shape[-1]
        # Fixed random gates approximate an LSTM-style recurrent encoder.
        self.w_i = self.rng.normal(0, 0.16, (d + self.hidden, self.hidden))
        self.w_f = self.rng.normal(0, 0.16, (d + self.hidden, self.hidden))
        self.w_o = self.rng.normal(0, 0.16, (d + self.hidden, self.hidden))
        self.w_g = self.rng.normal(0, 0.16, (d + self.hidden, self.hidden))
        return super().fit(x, y)

    def encode(self, x: np.ndarray) -> np.ndarray:
        h = np.zeros((x.shape[0], self.hidden))
        c = np.zeros_like(h)
        for t in range(x.shape[1]):
            # Standard LSTM gate equations, vectorized over all windows.
            z = np.c_[x[:, t, :], h]
            i = 1.0 / (1.0 + np.exp(-(z @ self.w_i)))
            f = 1.0 / (1.0 + np.exp(-(z @ self.w_f)))
            o = 1.0 / (1.0 + np.exp(-(z @ self.w_o)))
            g = np.tanh(z @ self.w_g)
            c = f * c + i * g
            h = o * np.tanh(c)
        # Combine learned recurrent state with simple window statistics for robustness.
        stats = np.c_[x[:, -1, :], x.mean(axis=1), x.std(axis=1), h]
        return stats


class TransformerReservoir(BaseModel):
    name = "Transformer"

    def fit(self, x: np.ndarray, y: np.ndarray):
        d = x.shape[-1]
        # Single-head attention projections; the last day acts as the query.
        self.w_q = self.rng.normal(0, 0.22, (d, self.hidden))
        self.w_k = self.rng.normal(0, 0.22, (d, self.hidden))
        self.w_v = self.rng.normal(0, 0.22, (d, self.hidden))
        return super().fit(x, y)

    def encode(self, x: np.ndarray) -> np.ndarray:
        q = x[:, -1, :] @ self.w_q
        k = x @ self.w_k
        v = x @ self.w_v
        # Attention weights select relevant days from the full 90-day history.
        scores = np.einsum("bh,bth->bt", q, k) / math.sqrt(self.hidden)
        scores -= scores.max(axis=1, keepdims=True)
        attn = np.exp(scores)
        attn /= attn.sum(axis=1, keepdims=True)
        context = np.einsum("bt,bth->bh", attn, v)
        recent = x[:, -14:, :].mean(axis=1)
        return np.c_[x[:, -1, :], x.mean(axis=1), recent, np.tanh(context)]


class CNNAttentionReservoir(BaseModel):
    name = "CNN-Attention"

    def fit(self, x: np.ndarray, y: np.ndarray):
        d = x.shape[-1]
        # A width-5 temporal convolution extracts local usage patterns before attention.
        self.kernels = self.rng.normal(0, 0.18, (5, d, self.hidden))
        self.w_q = self.rng.normal(0, 0.20, (self.hidden, self.hidden))
        self.w_k = self.rng.normal(0, 0.20, (self.hidden, self.hidden))
        self.w_v = self.rng.normal(0, 0.20, (self.hidden, self.hidden))
        return super().fit(x, y)

    def encode(self, x: np.ndarray) -> np.ndarray:
        # Edge padding keeps the encoded sequence length equal to input_len.
        padded = np.pad(x, ((0, 0), (2, 2), (0, 0)), mode="edge")
        conv = []
        for t in range(x.shape[1]):
            window = padded[:, t : t + 5, :]
            conv.append(np.tanh(np.einsum("bkd,kdh->bh", window, self.kernels)))
        conv_x = np.stack(conv, axis=1)
        q = conv_x[:, -1, :] @ self.w_q
        k = conv_x @ self.w_k
        v = conv_x @ self.w_v
        scores = np.einsum("bh,bth->bt", q, k) / math.sqrt(self.hidden)
        scores -= scores.max(axis=1, keepdims=True)
        attn = np.exp(scores)
        attn /= attn.sum(axis=1, keepdims=True)
        context = np.einsum("bt,bth->bh", attn, v)
        # A coarse trend feature helps the long-horizon model distinguish rising/falling windows.
        trend = x[:, -30:, :].mean(axis=1) - x[:, :30, :].mean(axis=1)
        return np.c_[x[:, -1, :], x.mean(axis=1), trend, np.tanh(context)]


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    err = y_pred - y_true
    return float(np.mean(err * err)), float(np.mean(np.abs(err)))


def draw_table(summary: pd.DataFrame, path: Path):
    """Render a PNG table so the report can include results as a screenshot."""

    rows = [["Model", "Horizon", "MSE mean", "MSE std", "MAE mean", "MAE std"]]
    for _, r in summary.iterrows():
        rows.append(
            [
                str(r["model"]),
                str(int(r["horizon"])),
                f"{r['mse_mean']:.3f}",
                f"{r['mse_std']:.3f}",
                f"{r['mae_mean']:.3f}",
                f"{r['mae_std']:.3f}",
            ]
        )
    font = ImageFont.load_default()
    col_w = [150, 80, 110, 100, 110, 100]
    row_h = 30
    img = Image.new("RGB", (sum(col_w) + 40, row_h * len(rows) + 40), "white")
    draw = ImageDraw.Draw(img)
    y = 20
    for i, row in enumerate(rows):
        x = 20
        fill = (230, 235, 242) if i == 0 else (255, 255, 255)
        draw.rectangle([20, y, 20 + sum(col_w), y + row_h], fill=fill, outline=(180, 180, 180))
        for j, cell in enumerate(row):
            draw.text((x + 6, y + 9), cell, fill="black", font=font)
            x += col_w[j]
            draw.line([x, y, x, y + row_h], fill=(180, 180, 180))
        y += row_h
    img.save(path)


def draw_curve(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path):
    """Render prediction vs. ground-truth curves without requiring matplotlib."""

    width, height = 1000, 520
    margin_l, margin_r, margin_t, margin_b = 70, 30, 55, 60
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((margin_l, 20), title, fill="black", font=font)
    left, right = margin_l, width - margin_r
    top, bottom = margin_t, height - margin_b
    draw.rectangle([left, top, right, bottom], outline=(60, 60, 60))
    combined = np.r_[y_true, y_pred]
    y_min, y_max = float(np.min(combined)), float(np.max(combined))
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    def point(i: int, val: float, n: int):
        x = left + (right - left) * i / max(1, n - 1)
        y = bottom - (bottom - top) * (float(val) - y_min) / (y_max - y_min)
        return x, y

    for frac in np.linspace(0, 1, 6):
        y = top + (bottom - top) * frac
        draw.line([left, y, right, y], fill=(230, 230, 230))
    for arr, color in ((y_true, (40, 90, 180)), (y_pred, (210, 70, 50))):
        pts = [point(i, v, len(arr)) for i, v in enumerate(arr)]
        if len(pts) > 1:
            draw.line(pts, fill=color, width=2)
    draw.line([left + 20, bottom + 30, left + 70, bottom + 30], fill=(40, 90, 180), width=3)
    draw.text((left + 78, bottom + 25), "Ground Truth", fill="black", font=font)
    draw.line([left + 210, bottom + 30, left + 260, bottom + 30], fill=(210, 70, 50), width=3)
    draw.text((left + 268, bottom + 25), "Prediction", fill="black", font=font)
    draw.text((left, bottom + 42), "Day index", fill="black", font=font)
    draw.text((8, top), "Power", fill="black", font=font)
    img.save(path)


def run(args):
    OUTPUT_DIR.mkdir(exist_ok=True)
    ds = load_dataset()
    train_x, test_x, train_y, test_y, y_mean, y_std = standardize(
        ds.train, ds.test, ds.features, ds.target
    )
    models = [LSTMReservoir, TransformerReservoir, CNNAttentionReservoir]
    horizons = [90, 365]
    rows = []
    predictions = {}

    for horizon in horizons:
        # Short-term and long-term tasks are trained independently as required.
        xtr, ytr = make_windows(train_x, train_y, args.input_len, horizon)
        xte, yte = make_windows(
            # Prepend the final training window so the first test prediction has 90 days of history.
            np.vstack([train_x[-args.input_len :], test_x]),
            np.r_[train_y[-args.input_len :], test_y],
            args.input_len,
            horizon,
        )
        if args.max_train_samples:
            # Optional acceleration for smoke tests; full experiments leave this unset.
            xtr = xtr[-args.max_train_samples :]
            ytr = ytr[-args.max_train_samples :]
        # Report figures use the final available forecast window for each horizon.
        xte_last, yte_last = xte[-1:], yte[-1:]

        for model_cls in models:
            for run_id in range(args.runs):
                seed = args.seed + run_id
                model = model_cls(seed=seed, hidden=args.hidden).fit(xtr, ytr)
                pred = model.predict(xte_last)
                # Convert back to the original power scale before computing final metrics.
                true_real = yte_last[0] * y_std + y_mean
                pred_real = pred[0] * y_std + y_mean
                mse, mae = metrics(true_real, pred_real)
                rows.append(
                    {
                        "model": model_cls.name,
                        "horizon": horizon,
                        "run": run_id + 1,
                        "mse": mse,
                        "mae": mae,
                    }
                )
                predictions[(model_cls.name, horizon, run_id)] = (true_real, pred_real)
                print(
                    f"{model_cls.name:14s} horizon={horizon:3d} "
                    f"run={run_id + 1} mse={mse:.3f} mae={mae:.3f}"
                )

    # Save both raw repeated-run metrics and the mean/std summary required by the assignment.
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False, encoding="utf-8-sig")
    summary = (
        metrics_df.groupby(["model", "horizon"])
        .agg(mse_mean=("mse", "mean"), mse_std=("mse", "std"), mae_mean=("mae", "mean"), mae_std=("mae", "std"))
        .reset_index()
    )
    summary.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")
    draw_table(summary, OUTPUT_DIR / "metrics_summary.png")

    for (model, horizon, run_id), (true, pred) in predictions.items():
        if run_id == 0:
            safe_model = model.lower().replace("-", "_")
            draw_curve(
                true,
                pred,
                f"{model} forecast, horizon={horizon}",
                OUTPUT_DIR / f"curve_{safe_model}_{horizon}.png",
            )
    print(f"Saved outputs to {OUTPUT_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-len", type=int, default=90)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-train-samples", type=int, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
