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

Put the course CSV files in `data/` or the project root:

- `train.csv`
- `test.csv` or `tes.csv`

If the course files are not present, the script attempts to download the UCI Individual household electric power consumption dataset and creates a time-ordered split automatically.

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

## GitHub

After creating the remote repository, update the link in `report/report.tex`.
