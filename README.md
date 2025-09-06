# qumula-trader

Quick Multi Agent Trader - FYP Research

## Objective

Explore whether introducing multi-agentic generative AI in low timeframe trading is feasible, and is able to generate alpha that beats single/zero-agent frameworks

## Setup

```sh
python -m venv venv
source ./venv/Scripts/activate

python main.py
```

## Results

```sh
Multi-Agent Crypto Trading Backtest
Generating sample data...
2025-09-07 01:00:28,197 - INFO - Generated data - Price range: $42659.26 - $52022.65
2025-09-07 01:00:28,197 - INFO - Max price change: 21.95%
Generated 720 data points from 2025-08-08 01:00:28.159103 to 2025-09-07 00:00:28.159103
Price range: $42659.26 - $52022.65
Running backtest...
2025-09-07 01:00:28,197 - INFO - Starting backtest with 720 data points
2025-09-07 01:00:28,969 - INFO - Processing data point 100/720
2025-09-07 01:00:30,249 - INFO - Processing data point 200/720
2025-09-07 01:00:31,794 - INFO - Processing data point 300/720
2025-09-07 01:00:33,370 - INFO - Processing data point 400/720
2025-09-07 01:00:34,945 - INFO - Processing data point 500/720
2025-09-07 01:00:36,497 - INFO - Processing data point 600/720
2025-09-07 01:00:38,046 - INFO - Processing data point 700/720
2025-09-07 01:00:38,360 - INFO - Backtest completed

==================================================
BACKTEST RESULTS
==================================================
Initial Balance: $10,000.00
Final Balance: $10,000.00
Total Return: 0.00%
Buy & Hold Return: 4.33%
Excess Return: -4.33%
Sharpe Ratio: 0.000
Max Drawdown: 0.00%
Volatility (Annual): 0.00%
Number of Trades: 0
Average Latency: 14.9ms
Max Latency: 34.2ms
==================================================

No trades executed during backtest period

Backtest completed successfully!
Final portfolio balance: $10000.00
No open positions

Diagnostics:
  Data points processed: 672
  Final BTC price: $46946.34
  Price change: 4.33%
```
