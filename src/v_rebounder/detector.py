from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class VReboundConfig:
    # Drop detection
    drop_threshold_pct: float = 3.0
    drop_window_bars: int = 5
    atr_drop_multiplier: float = 2.0

    # Recovery detection
    recovery_threshold_pct: float = 1.5
    recovery_window_bars: int = 3
    max_recovery_bars: int = 5

    # Volume confirmation
    volume_spike_mult: float = 2.0
    volume_decline_ratio: float = 0.7

    # RSI divergence
    rsi_oversold: float = 35.0
    rsi_divergence_min: float = 5.0
    divergence_lookback: int = 20

    # Dead cat bounce filter
    max_retest_count: int = 5
    retest_window: int = 10
    retest_tolerance: float = 1.01

    # Flags
    use_dynamic_threshold: bool = True
    use_rsi_divergence: bool = True
    use_volume_exhaustion: bool = True
    use_dead_cat_filter: bool = True


class VReboundDetector:
    def __init__(self, config: VReboundConfig | None = None):
        self.config = config or VReboundConfig()

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config

        df = self._add_swing_structure(df)
        df = self._add_drop_recovery(df)
        df = self._add_volume_analysis(df)

        if cfg.use_rsi_divergence and "rsi" in df.columns:
            df = self._add_rsi_divergence(df)

        if cfg.use_dead_cat_filter:
            df = self._add_dead_cat_filter(df)

        df = self._generate_signals(df)

        return df

    def _add_swing_structure(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config

        df = df.with_columns(
            pl.col("high")
            .rolling_max(window_size=cfg.drop_window_bars)
            .shift(1)
            .alias("swing_high"),
            pl.col("low")
            .rolling_min(window_size=cfg.drop_window_bars)
            .alias("swing_low"),
            pl.col("low")
            .rolling_min(window_size=cfg.divergence_lookback)
            .alias("swing_low_extended"),
        )

        return df

    def _add_drop_recovery(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config

        df = df.with_columns(
            ((pl.col("swing_high") - pl.col("low")) / pl.col("swing_high") * 100).alias(
                "drop_pct"
            ),
            ((pl.col("close") - pl.col("swing_low")) / pl.col("swing_low") * 100).alias(
                "recovery_pct"
            ),
        )

        if cfg.use_dynamic_threshold and "atr" in df.columns:
            # ATR-normalized drop: more robust than fixed %
            df = df.with_columns(
                ((pl.col("swing_high") - pl.col("low")) / pl.col("atr")).alias("atr_drop"),
                (pl.col("atr") / pl.col("close") * 100 * cfg.atr_drop_multiplier).alias(
                    "dynamic_threshold"
                ),
            )

        return df

    def _add_volume_analysis(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config

        if "volume_ma" not in df.columns:
            df = df.with_columns(
                pl.col("volume").rolling_mean(window_size=20).alias("volume_ma"),
            )

        df = df.with_columns(
            (pl.col("volume") / pl.col("volume_ma")).alias("volume_ratio"),
            (pl.col("volume") >= pl.col("volume_ma") * cfg.volume_spike_mult).alias(
                "volume_spike"
            ),
        )

        if cfg.use_volume_exhaustion:
            # Volume exhaustion: spike followed by decline = capitulation complete
            df = df.with_columns(
                (
                    pl.col("volume").shift(1) >= pl.col("volume_ma").shift(1) * cfg.volume_spike_mult
                ).alias("prev_volume_spike"),
                (pl.col("volume") < pl.col("volume").shift(1) * cfg.volume_decline_ratio).alias(
                    "volume_declining"
                ),
            )

            df = df.with_columns(
                (pl.col("prev_volume_spike") & pl.col("volume_declining")).alias(
                    "volume_exhaustion"
                ),
            )

        return df

    def _add_rsi_divergence(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config

        # RSI at swing low (lookback window)
        df = df.with_columns(
            pl.col("rsi").rolling_min(window_size=cfg.divergence_lookback).alias("rsi_at_low"),
        )

        # Bullish divergence: price near/below swing low, RSI higher than at swing low
        df = df.with_columns(
            (
                (pl.col("low") <= pl.col("swing_low_extended") * 1.01)
                & (pl.col("rsi") > pl.col("rsi_at_low") + cfg.rsi_divergence_min)
                & (pl.col("rsi") <= cfg.rsi_oversold + 10)
            ).alias("rsi_bullish_divergence"),
        )

        # RSI recovering (not still declining)
        df = df.with_columns(
            (pl.col("rsi") > pl.col("rsi").shift(1)).alias("rsi_recovering"),
        )

        return df

    def _add_dead_cat_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config

        # Count re-tests of swing low in recent window
        df = df.with_columns(
            (pl.col("low") <= pl.col("swing_low") * cfg.retest_tolerance)
            .rolling_sum(window_size=cfg.retest_window)
            .alias("retest_count"),
        )

        # Dead cat bounce: multiple retests = weak support
        df = df.with_columns(
            (pl.col("retest_count") <= cfg.max_retest_count).alias("no_excessive_retest"),
        )

        # RSI still declining = no real reversal
        if "rsi" in df.columns:
            df = df.with_columns(
                (
                    pl.col("rsi").rolling_mean(window_size=3)
                    > pl.col("rsi").rolling_mean(window_size=3).shift(3)
                ).alias("rsi_trend_up"),
            )
        else:
            df = df.with_columns(pl.lit(True).alias("rsi_trend_up"))

        return df

    def _generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        cfg = self.config

        # Base conditions
        if cfg.use_dynamic_threshold and "dynamic_threshold" in df.columns:
            drop_condition = pl.col("drop_pct") >= pl.max_horizontal(
                pl.lit(cfg.drop_threshold_pct), pl.col("dynamic_threshold")
            )
        else:
            drop_condition = pl.col("drop_pct") >= cfg.drop_threshold_pct

        recovery_condition = pl.col("recovery_pct") >= cfg.recovery_threshold_pct

        # Volume conditions - spike OR exhaustion pattern
        if cfg.use_volume_exhaustion and "volume_exhaustion" in df.columns:
            volume_condition = pl.col("volume_spike") | pl.col("volume_exhaustion")
        else:
            volume_condition = pl.col("volume_spike")

        # Basic signal: drop + recovery + volume
        basic_signal = drop_condition & recovery_condition & volume_condition

        # RSI filter: must be oversold OR recovering (relaxed condition)
        if "rsi" in df.columns and "rsi_recovering" in df.columns:
            rsi_favorable = (pl.col("rsi") <= cfg.rsi_oversold + 15) | pl.col("rsi_recovering")
        elif "rsi" in df.columns:
            rsi_favorable = pl.col("rsi") <= cfg.rsi_oversold + 15
        else:
            rsi_favorable = pl.lit(True)

        # Enhanced signal: basic + RSI favorable
        enhanced_signal = basic_signal & rsi_favorable

        # Dead cat filter (optional, relaxed)
        if cfg.use_dead_cat_filter and "no_excessive_retest" in df.columns:
            filtered_signal = enhanced_signal & pl.col("no_excessive_retest")
        else:
            filtered_signal = enhanced_signal

        df = df.with_columns(
            basic_signal.alias("v_rebound_basic"),
            enhanced_signal.alias("v_rebound_enhanced"),
            filtered_signal.alias("v_rebound_signal"),
        )

        return df

    def detect_with_candlestick(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self.detect(df)

        df = df.with_columns(
            (pl.col("close") - pl.col("open")).alias("body"),
            (pl.col("high") - pl.max_horizontal("open", "close")).alias("upper_shadow"),
            (pl.min_horizontal("open", "close") - pl.col("low")).alias("lower_shadow"),
        )

        body_threshold = (pl.col("high") - pl.col("low")) * 0.3

        # Hammer: small body at top, long lower shadow (2x body)
        hammer_condition = (
            (pl.col("body").abs() <= body_threshold)
            & (pl.col("lower_shadow") >= 2 * pl.col("body").abs())
            & (pl.col("upper_shadow") <= pl.col("body").abs() * 0.5)
        )

        # Bullish engulfing: current bullish candle covers previous bearish
        bullish_engulfing = (
            (pl.col("close") > pl.col("open"))
            & (pl.col("close").shift(1) < pl.col("open").shift(1))
            & (pl.col("open") <= pl.col("close").shift(1))
            & (pl.col("close") >= pl.col("open").shift(1))
        )

        # Doji: very small body (indecision at bottom)
        doji_condition = pl.col("body").abs() <= (pl.col("high") - pl.col("low")) * 0.1

        df = df.with_columns(
            hammer_condition.alias("hammer"),
            bullish_engulfing.alias("bullish_engulfing"),
            doji_condition.alias("doji"),
        )

        df = df.with_columns(
            (
                pl.col("v_rebound_signal")
                & (pl.col("hammer") | pl.col("bullish_engulfing") | pl.col("doji"))
            ).alias("v_rebound_with_pattern"),
        )

        return df.drop(["body", "upper_shadow", "lower_shadow"])


# Legacy compatibility
class VReboundDetectorLegacy:
    """Original simple detector for baseline comparison."""

    def __init__(
        self,
        drop_threshold_pct: float = 3.0,
        drop_window_bars: int = 5,
        recovery_threshold_pct: float = 1.5,
        volume_spike_mult: float = 2.0,
    ):
        self.drop_threshold_pct = drop_threshold_pct
        self.drop_window_bars = drop_window_bars
        self.recovery_threshold_pct = recovery_threshold_pct
        self.volume_spike_mult = volume_spike_mult

    def detect(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col("high")
            .rolling_max(window_size=self.drop_window_bars)
            .shift(1)
            .alias("rolling_high"),
            pl.col("low")
            .rolling_min(window_size=self.drop_window_bars)
            .alias("rolling_low"),
        )

        df = df.with_columns(
            ((pl.col("rolling_high") - pl.col("low")) / pl.col("rolling_high") * 100).alias(
                "drop_pct"
            ),
            ((pl.col("close") - pl.col("rolling_low")) / pl.col("rolling_low") * 100).alias(
                "recovery_pct"
            ),
        )

        drop_cond = pl.col("drop_pct") >= self.drop_threshold_pct
        recovery_cond = pl.col("recovery_pct") >= self.recovery_threshold_pct

        if "volume_ma" in df.columns:
            volume_cond = pl.col("volume") >= pl.col("volume_ma") * self.volume_spike_mult
        else:
            volume_cond = pl.lit(True)

        df = df.with_columns(
            (drop_cond & recovery_cond & volume_cond).alias("v_rebound_signal"),
        )

        return df
