# Machine Learning Quant Stock Selection

## Overview

This project uses machine learning to predict stock price movement direction for CSI 500 constituents and builds quantitative stock selection strategies based on model outputs.

## Research Question

Can machine learning models improve quantitative stock selection performance compared with directly tracking the CSI 500 Index?

## Methods

- Used CSI 500 constituent data from August 16, 2024 to November 29, 2024.
- Built labels based on future one-week returns and price movement direction.
- Used 15 indicators, including technical and financial factors.
- Applied PCA for dimensionality reduction.
- Compared SVM, logistic regression, KNN, and random forest models.
- Evaluated models using accuracy, precision, recall, and F1 score.
- Constructed and backtested stock selection strategies based on model predictions.

## Results

- Logistic regression achieved the highest accuracy and precision in the model comparison.
- SVM achieved the highest recall and F1 score.
- Reported strategy returns included 51.26% for logistic regression, 52.06% for random forest, and 44.25% for KNN.
- All machine learning strategies discussed in the report outperformed the CSI 500 Index return of -0.34% during the backtest period.

## Files

- `ml_quant_stock_selection.ipynb`: Sanitized Jupyter Notebook.
- `report.pdf`: Full project report.
- `data/README.md`: Data availability note.

## Resume Bullet

Built a CSI 500 machine learning stock selection system using PCA and four classifiers; backtests showed ML strategies outperformed the CSI 500 benchmark, with random forest returning 52.06% versus -0.34%.
