# Long-Term Time Series Forecasting Benchmark

## Overview

This project benchmarks machine learning and deep learning models for long-term multivariate time series forecasting. It compares four representative forecasting paradigms: DLinear, LightGBM, LSTM, and PatchTST.

## Research Question

How do traditional, machine-learning-based, recurrent neural network, and Transformer-based forecasting models perform under different datasets and prediction horizons?

## Methods

- Evaluated ETTm1, Electricity, and Weather benchmark datasets.
- Compared DLinear, LightGBM, LSTM, and PatchTST.
- Tested four prediction horizons: 96, 192, 336, and 720.
- Reported MAE, RMSE, MSE, MAPE, dataset size, and train/validation/test split statistics.
- Adapted and reproduced experiments from Time-Series-Library where appropriate.
- Used lagged supervised learning features for LightGBM to align feature-based modeling with neural sequence forecasting tasks.

## Results

- LightGBM achieved the strongest overall numerical performance across the benchmark.
- PatchTST was the strongest deep learning model and captured long-range temporal structure better than the recurrent baseline.
- LSTM provided a simple recurrent neural baseline but was less stable under longer horizons and harder datasets.
- DLinear served as a trend-oriented reference model and showed clear smoothing behavior.
- Forecasting difficulty generally increased as the prediction horizon expanded, but the pattern differed by model family and dataset.

## Files

- `report.pdf`: Full project report.
- `code/Linearmodel.py`: Linear forecasting baseline.
- `code/Lightgbm.ipynb`: LightGBM forecasting experiments.
- `code/LSTM.py`: LSTM result analysis utilities.
- `code/patchtst_runner.py`: PatchTST runner and evaluation pipeline.
- `data/README.md`: Dataset availability note.

## Resume Bullet

Benchmarked DLinear, LightGBM, LSTM, and PatchTST for long-term multivariate forecasting on ETTm1, Electricity, and Weather datasets across four prediction horizons; found LightGBM strongest overall and PatchTST strongest among deep learning models.
