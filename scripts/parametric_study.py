#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v_rebounder.data_loader import DataConfig, add_indicators, load_btc_data, resample_ohlcv
from v_rebounder.parametric import ParametricConfig, results_to_dataframe, run_parametric_study


def main() -> None:
    config = DataConfig(symbol="BTCUSDT")

    print("Loading BTC data...")
    df = load_btc_data(config)
    print(f"Loaded {len(df):,} rows")

    timeframe = "5m"
    print(f"\nResampling to {timeframe}...")
    df_tf = resample_ohlcv(df, timeframe)
    print(f"Resampled to {len(df_tf):,} rows")

    print("Adding indicators...")
    df_tf = add_indicators(df_tf)

    param_config = ParametricConfig(
        drop_threshold_pct=[2.0, 3.0, 4.0, 5.0, 6.0],
        drop_window_bars=[3, 5, 7, 10],
        recovery_threshold_pct=[0.5, 1.0, 1.5, 2.0],
        recovery_window_bars=[2, 3, 5],
        volume_spike_mult=[1.5, 2.0, 2.5],
        atr_multiplier=[1.5, 2.0, 2.5],
        take_profit_pct=[2.0, 3.0, 4.0, 5.0],
        stop_loss_pct=[1.0, 1.5, 2.0, 2.5],
    )

    total_combinations = (
        len(param_config.drop_threshold_pct)
        * len(param_config.drop_window_bars)
        * len(param_config.recovery_threshold_pct)
        * len(param_config.recovery_window_bars)
        * len(param_config.volume_spike_mult)
        * len(param_config.atr_multiplier)
        * len(param_config.take_profit_pct)
        * len(param_config.stop_loss_pct)
    )
    print(f"\nRunning parametric study ({total_combinations:,} combinations)...")

    results = run_parametric_study(
        df_tf,
        param_config,
        signal_column="v_rebound_signal",
        initial_capital=10000.0,
        top_n=20,
        min_trades=30,
    )

    print(f"\n{'='*80}")
    print("TOP 20 PARAMETER COMBINATIONS (by Calmar Ratio)")
    print(f"{'='*80}\n")

    for i, r in enumerate(results[:20], 1):
        print(f"Rank {i}:")
        print(f"  Params: drop={r.params['drop_threshold_pct']}%, window={r.params['drop_window_bars']}, "
              f"recovery={r.params['recovery_threshold_pct']}%, vol_mult={r.params['volume_spike_mult']}")
        print(f"  TP={r.params['take_profit_pct']}%, SL={r.params['stop_loss_pct']}%")
        print(f"  Return: {r.result.total_return_pct:.2f}% | MaxDD: {r.result.max_drawdown_pct:.2f}% | "
              f"WinRate: {r.result.win_rate:.2f}% | Trades: {r.result.total_trades}")
        print(f"  Calmar: {r.score:.2f} | Sharpe: {r.result.sharpe_ratio:.2f} | PF: {r.result.profit_factor:.2f}")
        print()

    results_df = results_to_dataframe(results)
    output_path = Path(__file__).parent.parent / "pages" / "parametric_results_5m.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.write_csv(output_path)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
