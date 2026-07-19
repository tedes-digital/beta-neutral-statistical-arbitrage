# Beta-Neutral Statistical Arbitrage Based on Cointegration

This repository contains the Python implementation and experimental results developed as part of the bachelor thesis:

**Beta-Neutral Statistical Arbitrage Based on Cointegration**

## Project overview

The project implements a statistical arbitrage strategy based on:

- selection of non-stationary stock price series;
- Engle–Granger cointegration testing;
- calculation of beta-neutral portfolio weights;
- rolling Z-score signal generation;
- simulation of trade execution;
- transaction costs and portfolio constraints;
- evaluation of profitability.

## Data

The analysis uses historical stock price data downloaded with the `yfinance` library.

## Main libraries

- pandas
- requests
- statsmodels
- yfinance

## Repository contents

The repository contains:

- Python source code;
- downloaded stock price data;
- generated trading signals;
- trading simulation results;
- summary CSV tables.

## Installation

```bash
pip install -r requirements.txt
```

## Author

Artem Chernukha

Bachelor thesis project, Czech Technical University in Prague.