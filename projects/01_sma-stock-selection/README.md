# SMA Stock Selection Strategy

## Overview

This project studies how to select A-share stocks suitable for a Simple Moving Average (SMA) trading strategy by combining fundamental factor screening and technical indicator scoring.

## Research Question

Can stocks with stronger fundamentals and favorable technical signals produce better SMA backtesting results than randomly selected stocks?

## Methods

- Collected and processed A-share market and financial data.
- Built a fundamental factor scoring framework covering profitability, operating efficiency, financial structure, and valuation.
- Applied winsorization and quartile scoring to reduce outlier impact.
- Evaluated technical indicators including SMA, MACD, RSI, MFI, APO, WILLR, ATR, and Bollinger Bands.
- Ran SMA strategy backtests and compared selected stocks with randomly selected stocks.

## Results

- Selected sample stocks included Wuliangye, Shanxi Fen Wine, and Zhejiang Meida.
- Reported annualized returns were 120.0%, 180.6%, and 20.9% respectively.
- Mean annualized return of selected stocks was 107.17%.
- Mean annualized return of randomly selected stocks was 2.53%, with only 38% exceeding the risk-free-rate benchmark.

## Files

- `sma_stock_selection.ipynb`: Sanitized Jupyter Notebook.
- `report.pdf`: Full project report.
- `mindmap.png`: Research workflow map.

## Resume Bullet

Developed an A-share SMA stock selection framework combining fundamental factor scoring and technical indicators; backtesting showed selected stocks achieved a 107.17% mean annualized return versus 2.53% for random selection.
