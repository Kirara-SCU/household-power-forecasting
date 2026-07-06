# Household Power Forecasting

This project implements the machine learning final assignment for multivariate household electric power forecasting.

It trains and evaluates three methods:

1. Fully trainable LSTM forecaster
2. Fully trainable Transformer encoder forecaster
3. Proposed CNN-Transformer hybrid forecaster

For each method, the script separately trains short-term and long-term forecasters:

- input window: 90 days
- short horizon: 90 days
- long horizon: 365 days
- metrics: MSE and MAE
- repeated experiments: 5 seeds, reporting mean and standard deviation

## Data

The script downloads and uses the UCI Individual household electric power consumption dataset by default.

## Run

```powershell
python src/run_experiments.py --runs 5
```

Use `--epochs` to increase or reduce the number of training epochs.

Useful faster smoke test:

```powershell
python src/run_experiments.py --runs 1 --epochs 1 --hidden 16 --layers 1 --max-train-samples 32
```

Outputs are written to `outputs/`:

- `metrics.csv`
- `metrics_summary.csv`
- `metrics_summary.png`
- prediction curve figures for each model and horizon
