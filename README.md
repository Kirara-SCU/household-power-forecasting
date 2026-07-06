# Household Power Forecasting

This project implements the machine learning final assignment for multivariate household electric power forecasting.

It trains and evaluates three methods:

1. LSTM-style recurrent reservoir encoder
2. Transformer-style self-attention encoder
3. Proposed CNN-attention hybrid encoder

For each method, the script separately trains short-term and long-term forecasters:

- input window: 90 days
- short horizon: 90 days
- long horizon: 365 days
- metrics: MSE and MAE
- repeated experiments: 5 seeds, reporting mean and standard deviation

## Data

Use the UCI Individual household electric power consumption dataset.

## Run

```powershell
python src/run_experiments.py --runs 5
```

Useful faster smoke test:

```powershell
python src/run_experiments.py --runs 1 --max-train-samples 80
```

Outputs are written to `outputs/`:

- `metrics.csv`
- `metrics_summary.csv`
- `metrics_summary.png`
- prediction curve figures for each model and horizon
