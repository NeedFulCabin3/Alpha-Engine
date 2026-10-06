from dataclasses import dataclass
from datetime import datetime, timezone
import math
import time
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# 1. DATA MODELS & CONFIGURATION
# ----------------------------------------------------------------------
@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    side: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float


@dataclass
class BacktestResult:
    total_return_pct: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    equity_curve: pd.Series
    trades: List[Trade]


# ----------------------------------------------------------------------
# 2. TECHNICAL INDICATORS & STRATEGY SIGNALS
# ----------------------------------------------------------------------
class Strategy:
    """Calculates technical indicators and generates trading signals."""

    @staticmethod
    def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Exponential moving average for smoothing
        avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @classmethod
    def generate_signals(
        cls,  # <--- Add cls as the first parameter
        df: pd.DataFrame,
        fast_ma: int = 10,
        slow_ma: int = 30,
        rsi_period: int = 14,
        rsi_overbought: float = 70.0,
        rsi_oversold: float = 30.0,
    ) -> pd.DataFrame:
        data = df.copy()
        
        data["rsi"] = cls.calculate_rsi(data["close"], period=rsi_period)
        # ...
        return data
    
        """
        Generates buy/sell signals based on SMA Crossover filtered by RSI.
        Signals:
           1  -> Long entry signal
          -1  -> Exit / Short entry signal
           0  -> Neutral / Hold
        """
        data = df.copy()

        # Calculate indicators
        data["fast_ma"] = data["close"].rolling(window=fast_ma).mean()
        data["slow_ma"] = data["close"].rolling(window=slow_ma).mean()
        data["rsi"] = Strategy.calculate_rsi(data["close"], period=rsi_period)

        # Signals
        data["signal"] = 0

        # Long Entry: Fast MA crosses above Slow MA AND RSI is not overbought
        long_condition = (
            (data["fast_ma"] > data["slow_ma"])
            & (data["fast_ma"].shift(1) <= data["slow_ma"].shift(1))
            & (data["rsi"] < rsi_overbought)
        )

        # Exit Signal: Fast MA crosses below Slow MA OR RSI reaches overbought threshold
        exit_condition = (
            (data["fast_ma"] < data["slow_ma"])
            & (data["fast_ma"].shift(1) >= data["slow_ma"].shift(1))
        ) | (data["rsi"] >= rsi_overbought)

        data.loc[long_condition, "signal"] = 1
        data.loc[exit_condition, "signal"] = -1

        return data


# ----------------------------------------------------------------------
# 3. BACKTESTING ENGINE & RISK METRICS
# ----------------------------------------------------------------------
class Engine:
    """Event-driven backtesting engine simulating order execution and portfolio tracking."""

    def __init__(
        self, initial_capital: float = 10000.0, commission: float = 0.001, slippage: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, df: pd.DataFrame) -> BacktestResult:
        capital = self.initial_capital
        position_size = 0.0
        entry_price = 0.0
        entry_time = None

        equity = []
        trades: List[Trade] = []

        for timestamp, row in df.iterrows():
            current_price = row["close"]
            signal = row["signal"]

            # Execute Signals
            if signal == 1 and position_size == 0:
                # Buy signal execution (incorporating slippage)
                entry_price = current_price * (1 + self.slippage)
                cost = capital * (1 - self.commission)
                position_size = cost / entry_price
                capital = 0.0
                entry_time = timestamp

            elif signal == -1 and position_size > 0:
                # Sell signal execution (incorporating slippage)
                exit_price = current_price * (1 - self.slippage)
                gross_proceeds = position_size * exit_price
                capital = gross_proceeds * (1 - self.commission)

                # Record trade stats
                pnl = capital - (position_size * entry_price)
                return_pct = (exit_price - entry_price) / entry_price
                trades.append(
                    Trade(
                        entry_time=entry_time,
                        exit_time=timestamp,
                        side="BUY",
                        entry_price=entry_price,
                        exit_price=exit_price,
                        size=position_size,
                        pnl=pnl,
                        return_pct=return_pct,
                    )
                )

                position_size = 0.0
                entry_price = 0.0
                entry_time = None

            # Track portfolio mark-to-market value
            current_portfolio_value = capital + (position_size * current_price)
            equity.append(current_portfolio_value)

        equity_series = pd.Series(equity, index=df.index)
        metrics = self._calculate_metrics(equity_series, trades)

        return BacktestResult(
            total_return_pct=metrics["total_return"],
            cagr=metrics["cagr"],
            sharpe_ratio=metrics["sharpe"],
            max_drawdown=metrics["max_drawdown"],
            win_rate=metrics["win_rate"],
            total_trades=len(trades),
            equity_curve=equity_series,
            trades=trades,
        )

    def _calculate_metrics(self, equity: pd.Series, trades: List[Trade]) -> Dict[str, float]:
        if equity.empty or len(equity) < 2:
            return {"total_return": 0.0, "cagr": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}

        # Returns & CAGR
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        total_days = (equity.index[-1] - equity.index[0]).days
        years = max(total_days / 365.25, 0.0027)  # Prevent zero division (min ~1 day)
        cagr = ((equity.iloc[-1] / self.initial_capital) ** (1 / years)) - 1 if equity.iloc[-1] > 0 else -1.0

        # Daily Returns & Sharpe Ratio (Risk-Free Rate = 0%)
        daily_returns = equity.pct_change().dropna()
        if daily_returns.std() != 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * math.sqrt(252)
        else:
            sharpe = 0.0

        # Maximum Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # Win Rate
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0.0

        return {
            "total_return": total_return,
            "cagr": cagr,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }


# ----------------------------------------------------------------------
# 4. LIVE TRADING EXECUTOR (CCXT INTEGRATION)
# ----------------------------------------------------------------------
class LiveTrader:
    """Polls CCXT market data and places orders on live/sandbox exchange APIs."""

    def __init__(
        self,
        exchange_id: str,
        symbol: str,
        api_key: str = "",
        secret: str = "",
        sandbox: bool = True,
    ):
        self.symbol = symbol
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class(
                {
                    "apiKey": api_key,
                    "secret": secret,
                    "enableRateLimit": True,
                }
            )
            if sandbox:
                self.exchange.set_sandbox_mode(True)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize exchange {exchange_id}: {e}")

    def fetch_ohlcv_data(self, timeframe: str = "1h", limit: int = 100) -> pd.DataFrame:
        """Fetches historical market candles and returns structured DataFrame."""
        try:
            raw_candles = self.exchange.fetch_ohlcv(self.symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(
                raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            print(f"[ERROR] Failed fetching data: {e}")
            return pd.DataFrame()

    def execute_live_cycle(self, timeframe: str = "1h"):
        """Performs a single strategy iteration: fetches data, evaluates signal, executes order."""
        print(f"[{datetime.now(timezone.utc).isoformat()}] Polling market context for {self.symbol}...")
        df = self.fetch_ohlcv_data(timeframe=timeframe, limit=100)

        if df.empty:
            return

        df = Strategy.generate_signals(df)
        latest_signal = df["signal"].iloc[-1]
        latest_price = df["close"].iloc[-1]

        print(f"Current Price: ${latest_price:,.2f} | Latest Signal: {latest_signal}")

        if latest_signal == 1:
            print(f"[ACTION] Signal 1 Received -> Placing BUY Market Order for {self.symbol}...")
            # Uncomment for real trading executions:
            # self.exchange.create_market_buy_order(self.symbol, amount=0.001)
        elif latest_signal == -1:
            print(f"[ACTION] Signal -1 Received -> Placing SELL Market Order for {self.symbol}...")
            # Uncomment for real trading executions:
            # self.exchange.create_market_sell_order(self.symbol, amount=0.001)
        else:
            print("[ACTION] No action taken. Holding position.")


# ----------------------------------------------------------------------
# 5. DEMO EXECUTION
# ----------------------------------------------------------------------
def generate_synthetic_data(days: int = 365) -> pd.DataFrame:
    """Generates synthetic price data (Geometric Brownian Motion) for strategy backtesting."""
    np.random.seed(42)
    timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=days * 24, freq="1h")
    returns = np.random.normal(0.0001, 0.015, size=len(timestamps))
    price_path = 100.0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": price_path * (1 - 0.001),
            "high": price_path * (1 + 0.004),
            "low": price_path * (1 - 0.004),
            "close": price_path,
            "volume": np.random.uniform(10, 500, size=len(timestamps)),
        },
        index=timestamps,
    )
    return df


if __name__ == "__main__":
    print("=== 1. BACKTESTING STRATEGY ===")
    data = generate_synthetic_data(days=365)
    processed_data = Strategy.generate_signals(data)

    engine = Engine(initial_capital=10000.0, commission=0.001, slippage=0.0005)
    results = engine.run(processed_data)

    print(f"Initial Balance:    ${engine.initial_capital:,.2f}")
    print(f"Final Equity:       ${results.equity_curve.iloc[-1]:,.2f}")
    print(f"Total Return:       {results.total_return_pct * 100:.2f}%")
    print(f"CAGR:               {results.cagr * 100:.2f}%")
    print(f"Sharpe Ratio:       {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown:       {results.max_drawdown * 100:.2f}%")
    print(f"Win Rate:           {results.win_rate * 100:.2f}%")
    print(f"Total Trades:       {results.total_trades}")

    print("\n=== 2. LIVE TRADER (TEST ENVIRONMENT) ===")
    trader = LiveTrader(
        exchange_id="binance",
        symbol="BTC/USDT",
        sandbox=True,
    )
    trader.execute_live_cycle(timeframe="1h")