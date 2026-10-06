# alpha-engine
> Event-driven algorithmic backtesting and live order execution suite for cryptocurrency trading strategies.

## Overview
Quantitative trading routines often face severe execution gaps between theoretical backtesting and live execution environments. Standard research workflows frequently ignore critical microstructural realities like transaction commissions, bid-ask slippage, and non-blocking market data polling.

`alpha-engine` bridges research and production by delivering a unified execution model. It pairs vectorized indicator calculations and signal generation with an event-driven backtesting engine that accounts for slippage and trading fees. The engine interfaces with CCXT sandbox and live exchange feeds to execute trade decisions directly against production endpoints.

## How It Works
The engine processes market state across three primary pipelines:

1. **Signal Processing Pipeline:** Historical or streaming OHLCV candle DataFrames pass into `Strategy.generate_signals()`. Exponential moving averages compute RSI across defined periods alongside dual simple moving averages (SMA). Buy (+1) and Sell/Exit (-1) triggers are evaluated vectorially based on crossover dynamics gated by overbought/oversold boundaries.
2. **Backtesting & Simulation Loop:** `Engine.run()` iterates sequentially across market timestamps. When entry or exit signals fire, orders execute incorporating fee deductions (`commission`) and unfavorable market fill adjustments (`slippage`). Equity values mark-to-market on every bar, producing a full time-series equity curve.
3. **Risk Profile Metrics Engine:** Upon backtest completion, `Engine._calculate_metrics()` computes portfolio metrics including Compound Annual Growth Rate (CAGR), maximum drawdown tracking via cumulative peak comparisons, win rates, and annualized Sharpe ratios using daily returns.
4. **Live Execution Gateway:** `LiveTrader` wraps CCXT instances, normalizes OHLCV payloads into internal Pandas DataFrames, evaluates live signals on the most recent candles, and dispatches market orders directly to exchange APIs.

## Key Features
- **Vectorized Technical Analysis:** Leverages Pandas and NumPy for exponential moving average (EMA) RSI smoothing and rolling window moving average calculations.
- **Realistic Backtesting Engine:** Models slippage penalties on fills (`price * (1 + slippage)`) and percentage-based commissions on proceeds.
- **Quantitative Risk Profiling:** Calculates Sharpe ratios, maximum drawdowns, CAGR (with division safety for short evaluation windows), and individual trade PnL distributions.
- **Exchange Agnostic Connectivity:** Integrates with the `ccxt` unified exchange client, supporting sandbox and paper trading environments.
- **Synthetic Market Data Generator:** Built-in Geometric Brownian Motion (GBM) simulation generates realistic stochastic price paths for offline testing.

## Tech Stack & Core Dependencies Breakdown
- **Python 3.10+:** Target runtime environment utilizing strong dataclass primitives and modern type hinting.
- **pandas (>= 2.0.0):** Data manipulation, rolling window statistics, and time-series index management.
- **numpy (>= 1.24.0):** Mathematical operations, exponential calculations, and synthetic data sampling.
- **ccxt (>= 4.0.0):** REST API abstraction wrapper providing unified interfaces across dozens of cryptocurrency exchanges.

## Environment & Web-Based Quick Start

### Running in GitHub Codespaces
1. Open this repository in GitHub and click **Code** > **Codespaces** > **Create codespace on main**.
2. Once the web terminal initializes, run the core script directly:
```bash
python main.py
```

### Local / Virtual Environment Setup

1. Clone the repository and navigate to the project directory:
```bash
git clone [https://github.com/your-username/alpha-engine.git](https://github.com/your-username/alpha-engine.git)
cd alpha-engine
```

2. Create and activate a Python virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install required dependencies:
```bash
pip install pandas numpy ccxt
```

4. Run the strategy engine:
```bash
python main.py
```

## Repository Structure

```bash
alpha-engine/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions automated linting & test suite
├── .gitignore                 # Python runtime, bytecode, and environment exclusions
├── LICENSE                    # MIT Open-Source License terms
├── README.md                  # Comprehensive technical documentation
└── main.py                    # Unified entrypoint containing indicators, engine, and live driver
```

## Roadmap

**[ ]AsyncIO Execution Refactoring:** Migrate LiveTrader polling logic to native asyncio loops utilizing ccxt.pro WebSockets for sub-second candle streaming and execution.

**[ ]Multi-Asset Portfolio Routing:** Expand Engine to track cross-margin allocation and multi-pair strategy allocation concurrently.

**[ ] Structured JSON Logging & Telemetry:** Replace inline print output with structured JSON logs (structlog) and Prometheus metrics for production monitoring.