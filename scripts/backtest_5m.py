#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v_rebounder.data_loader import DataConfig, add_indicators, load_btc_data, resample_ohlcv
from v_rebounder.detector import VReboundConfig, VReboundDetector
from v_rebounder.strategy import BacktestResult, TradeConfig, VReboundStrategy


def print_result(result: BacktestResult, timeframe: str) -> None:
    print(f"\n{'='*60}")
    print(f"V-REBOUNDER BASELINE BACKTEST - {timeframe}")
    print(f"{'='*60}")
    print(f"Total Return:          {result.total_return_pct:>10.2f}%")
    print(f"Max Drawdown:          {result.max_drawdown_pct:>10.2f}%")
    print(f"Win Rate:              {result.win_rate:>10.2f}%")
    print(f"Profit Factor:         {result.profit_factor:>10.2f}")
    print(f"Sharpe Ratio:          {result.sharpe_ratio:>10.2f}")
    print(f"Total Trades:          {result.total_trades:>10d}")
    print(f"Exp. Return/Trade:     {result.expected_return_per_trade:>10.2f}%")

    calmar = result.total_return_pct / result.max_drawdown_pct if result.max_drawdown_pct > 0 else 0
    print(f"Calmar Ratio:          {calmar:>10.2f}")
    print(f"{'='*60}\n")


def main() -> None:
    config = DataConfig(symbol="BTCUSDT")

    print("Loading BTC data...")
    try:
        df = load_btc_data(config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure data exists at /mnt/data/finance/cryptocurrency/BTCUSDT/")
        return

    print(f"Loaded {len(df):,} rows")

    timeframe = "5m"
    print(f"\nResampling to {timeframe}...")
    df_5m = resample_ohlcv(df, timeframe)
    print(f"Resampled to {len(df_5m):,} rows")

    print("Adding indicators...")
    df_5m = add_indicators(df_5m)

    detector_config = VReboundConfig(
        drop_threshold_pct=3.0,
        drop_window_bars=5,
        recovery_threshold_pct=1.5,
        recovery_window_bars=3,
        volume_spike_mult=2.0,
        atr_multiplier=2.0,
        rsi_oversold=35.0,
    )

    trade_config = TradeConfig(
        take_profit_pct=3.0,
        stop_loss_pct=2.0,
        half_close_enabled=True,
        half_close_pct=1.5,
        position_size_pct=20.0,
        leverage=10.0,
    )

    print("\nRunning V-Rebound detection...")
    detector = VReboundDetector(detector_config)
    df_signals = detector.detect(df_5m)

    signal_count = df_signals.filter("v_rebound_signal").height
    print(f"Detected {signal_count:,} V-Rebound signals")

    print("\nRunning backtest...")
    strategy = VReboundStrategy(trade_config)
    result = strategy.backtest(df_signals, "v_rebound_signal", initial_capital=10000.0)

    print_result(result, timeframe)

    print("\nSample trades (first 5):")
    for i, trade in enumerate(result.trades[:5]):
        print(f"  {i+1}. Entry: {trade.entry_price:.2f} -> Exit: {trade.exit_price:.2f} | PnL: {trade.pnl_pct:.2f}% | {trade.exit_reason}")


if __name__ == "__main__":
    main()
