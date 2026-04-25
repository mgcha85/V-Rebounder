from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import polars as pl


@dataclass
class TradeConfig:
    take_profit_pct: float = 3.0
    stop_loss_pct: float = 2.0
    half_close_enabled: bool = True
    half_close_pct: float = 1.5
    position_size_pct: float = 20.0
    leverage: float = 10.0
    maker_fee_pct: float = 0.02
    taker_fee_pct: float = 0.05
    slippage_pct: float = 0.05


@dataclass
class Trade:
    entry_time: str
    entry_price: float
    exit_time: str | None = None
    exit_price: float | None = None
    half_closed: bool = False
    half_close_price: float | None = None
    pnl_pct: float = 0.0
    status: Literal["open", "closed"] = "open"
    exit_reason: str | None = None


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_trades: int = 0
    expected_return_per_trade: float = 0.0
    alpha: float | None = None


class VReboundStrategy:
    def __init__(self, trade_config: TradeConfig | None = None):
        self.config = trade_config or TradeConfig()

    def backtest(
        self,
        df: pl.DataFrame,
        signal_column: str = "v_rebound_signal",
        initial_capital: float = 10000.0,
    ) -> BacktestResult:
        cfg = self.config
        trades: list[Trade] = []
        capital = initial_capital
        equity_curve = [capital]
        current_trade: Trade | None = None

        df_np = df.select(
            "open_time", "open", "high", "low", "close", signal_column
        ).to_numpy()

        for i in range(1, len(df_np)):
            row = df_np[i]
            time_str = str(row[0])
            open_price, high, low, close = row[1], row[2], row[3], row[4]
            signal = row[5]

            if current_trade is not None:
                entry = current_trade.entry_price

                # Check stop loss
                sl_price = entry * (1 - cfg.stop_loss_pct / 100)
                if low <= sl_price:
                    exit_price = sl_price * (1 - cfg.slippage_pct / 100)
                    pnl = self._calc_pnl(current_trade, exit_price, cfg)
                    current_trade.exit_time = time_str
                    current_trade.exit_price = exit_price
                    current_trade.pnl_pct = pnl
                    current_trade.status = "closed"
                    current_trade.exit_reason = "stop_loss"
                    trades.append(current_trade)
                    capital *= 1 + pnl / 100
                    current_trade = None
                    equity_curve.append(capital)
                    continue

                # Check half close
                if cfg.half_close_enabled and not current_trade.half_closed:
                    hc_price = entry * (1 + cfg.half_close_pct / 100)
                    if high >= hc_price:
                        current_trade.half_closed = True
                        current_trade.half_close_price = hc_price

                # Check take profit
                tp_price = entry * (1 + cfg.take_profit_pct / 100)
                if high >= tp_price:
                    exit_price = tp_price * (1 - cfg.slippage_pct / 100)
                    pnl = self._calc_pnl(current_trade, exit_price, cfg)
                    current_trade.exit_time = time_str
                    current_trade.exit_price = exit_price
                    current_trade.pnl_pct = pnl
                    current_trade.status = "closed"
                    current_trade.exit_reason = "take_profit"
                    trades.append(current_trade)
                    capital *= 1 + pnl / 100
                    current_trade = None
                    equity_curve.append(capital)
                    continue

                equity_curve.append(capital)

            elif signal:
                entry_price = close * (1 + cfg.slippage_pct / 100)
                current_trade = Trade(entry_time=time_str, entry_price=entry_price)
                equity_curve.append(capital)
            else:
                equity_curve.append(capital)

        # Close any open trade at end
        if current_trade is not None:
            exit_price = df_np[-1, 4]
            pnl = self._calc_pnl(current_trade, exit_price, cfg)
            current_trade.exit_time = str(df_np[-1, 0])
            current_trade.exit_price = exit_price
            current_trade.pnl_pct = pnl
            current_trade.status = "closed"
            current_trade.exit_reason = "end_of_data"
            trades.append(current_trade)
            capital *= 1 + pnl / 100

        return self._calc_metrics(trades, equity_curve, initial_capital)

    def _calc_pnl(self, trade: Trade, exit_price: float, cfg: TradeConfig) -> float:
        entry = trade.entry_price
        gross_pnl_pct = (exit_price - entry) / entry * 100

        fee_pct = cfg.taker_fee_pct * 2

        if trade.half_closed and trade.half_close_price is not None:
            hc_pnl = (trade.half_close_price - entry) / entry * 100 * 0.5
            remaining_pnl = gross_pnl_pct * 0.5
            gross_pnl_pct = hc_pnl + remaining_pnl
            fee_pct += cfg.taker_fee_pct

        position_pnl = gross_pnl_pct * cfg.leverage * (cfg.position_size_pct / 100)

        return position_pnl - fee_pct

    def _calc_metrics(
        self, trades: list[Trade], equity_curve: list[float], initial_capital: float
    ) -> BacktestResult:
        if not trades:
            return BacktestResult()

        pnls = [t.pnl_pct for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100

        # Max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        win_rate = len(wins) / len(trades) * 100 if trades else 0.0

        profit_factor = sum(wins) / sum(losses) if losses else float("inf")

        # Sharpe (simplified daily returns)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 12)  # 5m bars
        else:
            sharpe = 0.0

        expected_return = np.mean(pnls) if pnls else 0.0

        return BacktestResult(
            trades=trades,
            total_return_pct=total_return,
            max_drawdown_pct=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            total_trades=len(trades),
            expected_return_per_trade=expected_return,
        )
