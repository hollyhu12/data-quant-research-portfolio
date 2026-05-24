# Education Expenditure Econometrics

## Overview

This project uses econometric methods to study factors influencing regional education expenditure in Chinese cities.

## Research Question

Which economic and demographic variables are associated with regional education expenditure, and does the regression model pass robustness diagnostics?

## Methods

- Built a multiple regression model for education expenditure.
- Used city-level variables including GDP, natural growth rate, fiscal expenditure, compulsory education students, total students, education level, and residents' disposable income.
- Detected heteroskedasticity in the initial model and optimized it using a log transformation of the dependent variable.
- Conducted economic significance tests, goodness-of-fit analysis, t-tests, F-tests, VIF multicollinearity checks, BP/White heteroskedasticity tests, and Durbin-Watson autocorrelation testing.

## Results

- Final model reached an R-squared of 0.813.
- Durbin-Watson statistic was 1.770, indicating no strong autocorrelation.
- GDP, natural growth rate, education level, compulsory education student count, and disposable income were positively associated with education expenditure.
- Total student count and local fiscal spending showed negative coefficients in the optimized model.

## Files

- `report.pdf`: Full project report.

## Resume Bullet

Modeled Chinese regional education expenditure with multiple regression and diagnostic testing; optimized the model via log transformation and achieved an R-squared of 0.813 with no strong autocorrelation.
