# A-Share Communications Equipment Risk and Sentiment

## Overview

This project analyzes risk and market sentiment in the A-share communications equipment industry under the digital economy context.

## Research Question

Can stock-level risk indicators be transformed into a market sentiment proxy, and can that proxy help explain or forecast stock price volatility?

## Methods

- Collected risk indicators including turnover, volatility, beta, Sharpe ratio, and Value at Risk.
- Standardized indicators and used PCA to construct stock-level risk scores.
- Used variance in risk scores as a proxy for market sentiment dispersion.
- Applied Granger causality testing to examine the relationship between risk score variance and stock price changes.
- Built an ARIMA model to forecast risk score variance and sentiment trends.

## Results

- PCA-based risk scores provided an integrated view of stock-level risk.
- Granger causality testing supported a relationship between market sentiment variance and stock price changes.
- The selected ARIMA model was ARIMA(0,1,1).
- Model diagnostics indicated no significant autocorrelation, approximately normal residuals, and no significant heteroskedasticity.

## Files

- `report.pdf`: Full thesis-style report.

## Resume Bullet

Constructed a PCA-based risk scoring framework for A-share communications equipment stocks and linked risk score variance to market sentiment using Granger causality and ARIMA forecasting.
