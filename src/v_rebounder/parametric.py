from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import polars as pl
from tqdm import tqdm

from .data_loader import add_indicators
from .detector import VReboundConfig, VReboundDetector
from .strategy import BacktestResult, TradeConfig, VReboundStrategy


@dataclass
class ParametricConfig:
    drop_threshold_pct: list[float] | None = None
    drop_window_bars: list[int] | None = None
    recovery_threshold_pct: list[float] | None = None
    recovery_window_bars: list[int] | None = None
    volume_spike_mult: list[float] | None = None
    atr_multiplier: list[float] | None = None
    take_profit_pct: list[float] | None = None
    stop_loss_pct: list[float] | None = None

    def __post_init__(self) -> None:
        if self.drop_threshold_pct is None:
            self.drop_threshold_pct = [2.0, 3.0, 4.0, 5.0]
        if self.drop_window_bars is None:
            self.drop_window_bars = [3, 5, 7, 10]
        if self.recovery_threshold_pct is None:
            self.recovery_threshold_pct = [1.0, 1.5, 2.0, 3.0]
        if self.recovery_window_bars is None:
            self.recovery_window_bars = [2, 3, 5]
        if self.volume_spike_mult is None:
            self.volume_spike_mult = [1.5, 2.0, 2.5, 3.0]
        if self.atr_multiplier is None:
            self.atr_multiplier = [1.5, 2.0, 2.5, 3.0]
        if self.take_profit_pct is None:
            self.take_profit_pct = [2.0, 3.0, 4.0, 5.0]
        if self.stop_loss_pct is None:
            self.stop_loss_pct = [1.5, 2.0, 2.5, 3.0]


@dataclass
class ParametricResult:
    params: dict[str, Any]
    result: BacktestResult
    score: float


def run_parametric_study(
    df: pl.DataFrame,
    param_config: ParametricConfig | None = None,
    signal_column: str = "v_rebound_signal",
    initial_capital: float = 10000.0,
    top_n: int = 10,
    min_trades: int = 20,
) -> list[ParametricResult]:
    if param_config is None:
        param_config = ParametricConfig()

    param_grid = list(
        product(
            param_config.drop_threshold_pct,
            param_config.drop_window_bars,
            param_config.recovery_threshold_pct,
            param_config.recovery_window_bars,
            param_config.volume_spike_mult,
            param_config.atr_multiplier,
            param_config.take_profit_pct,
            param_config.stop_loss_pct,
        )
    )

    results: list[ParametricResult] = []

    for params in tqdm(param_grid, desc="Parametric Study"):
        (
            drop_thresh,
            drop_window,
            recovery_thresh,
            recovery_window,
            vol_mult,
            atr_mult,
            tp,
            sl,
        ) = params

        detector_config = VReboundConfig(
            drop_threshold_pct=drop_thresh,
            drop_window_bars=drop_window,
            recovery_threshold_pct=recovery_thresh,
            recovery_window_bars=recovery_window,
            volume_spike_mult=vol_mult,
            atr_multiplier=atr_mult,
        )

        trade_config = TradeConfig(
            take_profit_pct=tp,
            stop_loss_pct=sl,
        )

        detector = VReboundDetector(detector_config)
        strategy = VReboundStrategy(trade_config)

        df_signals = detector.detect(df)
        result = strategy.backtest(df_signals, signal_column, initial_capital)

        if result.total_trades < min_trades:
            continue

        # Calmar ratio as primary score
        calmar = (
            result.total_return_pct / result.max_drawdown_pct
            if result.max_drawdown_pct > 0
            else 0
        )

        param_dict = {
            "drop_threshold_pct": drop_thresh,
            "drop_window_bars": drop_window,
            "recovery_threshold_pct": recovery_thresh,
            "recovery_window_bars": recovery_window,
            "volume_spike_mult": vol_mult,
            "atr_multiplier": atr_mult,
            "take_profit_pct": tp,
            "stop_loss_pct": sl,
        }

        results.append(ParametricResult(params=param_dict, result=result, score=calmar))

    results.sort(key=lambda x: x.score, reverse=True)

    return results[:top_n]


def results_to_dataframe(results: list[ParametricResult]) -> pl.DataFrame:
    rows = []
    for r in results:
        row = {
            **r.params,
            "total_return_pct": r.result.total_return_pct,
            "max_drawdown_pct": r.result.max_drawdown_pct,
            "win_rate": r.result.win_rate,
            "profit_factor": r.result.profit_factor,
            "sharpe_ratio": r.result.sharpe_ratio,
            "total_trades": r.result.total_trades,
            "expected_return_per_trade": r.result.expected_return_per_trade,
            "calmar_ratio": r.score,
        }
        rows.append(row)

    return pl.DataFrame(rows)
