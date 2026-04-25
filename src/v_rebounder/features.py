from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass
class FeatureConfig:
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    momentum_period: int = 10
    roc_period: int = 12
    wma_periods: list[int] | None = None
    volume_ma_period: int = 20

    def __post_init__(self) -> None:
        if self.wma_periods is None:
            self.wma_periods = [5, 10, 20, 50]


def add_ml_features(df: pl.DataFrame, config: FeatureConfig | None = None) -> pl.DataFrame:
    if config is None:
        config = FeatureConfig()

    df = _add_price_features(df, config)
    df = _add_momentum_features(df, config)
    df = _add_volatility_features(df, config)
    df = _add_volume_features(df, config)
    df = _add_pattern_features(df)

    return df


def _add_price_features(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    for period in cfg.wma_periods:
        weights = np.arange(1, period + 1, dtype=np.float64)
        weights = weights / weights.sum()

        df = df.with_columns(
            pl.col("close")
            .rolling_map(lambda x: np.dot(x, weights[-len(x):]), window_size=period)
            .alias(f"wma_{period}"),
        )

    df = df.with_columns(
        (pl.col("close") / pl.col("wma_20") - 1).alias("price_wma20_ratio"),
        (pl.col("close") / pl.col("wma_50") - 1).alias("price_wma50_ratio"),
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low") + 1e-8)).alias(
            "hl_position"
        ),
        (pl.col("close") - pl.col("open")).alias("body"),
        ((pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 1e-8)).alias(
            "body_ratio"
        ),
    )

    return df


def _add_momentum_features(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    if "rsi" not in df.columns:
        price_change = pl.col("close") - pl.col("close").shift(1)
        gain = pl.when(price_change > 0).then(price_change).otherwise(0)
        loss = pl.when(price_change < 0).then(price_change.abs()).otherwise(0)

        df = df.with_columns(
            gain.rolling_mean(window_size=cfg.rsi_period).alias("avg_gain"),
            loss.rolling_mean(window_size=cfg.rsi_period).alias("avg_loss"),
        )

        df = df.with_columns(
            (100 - (100 / (1 + pl.col("avg_gain") / (pl.col("avg_loss") + 1e-8)))).alias("rsi"),
        )

        df = df.drop(["avg_gain", "avg_loss"])

    # MACD
    ema_fast = pl.col("close").ewm_mean(span=cfg.macd_fast, adjust=False)
    ema_slow = pl.col("close").ewm_mean(span=cfg.macd_slow, adjust=False)

    df = df.with_columns(
        (ema_fast - ema_slow).alias("macd"),
    )

    df = df.with_columns(
        pl.col("macd").ewm_mean(span=cfg.macd_signal, adjust=False).alias("macd_signal"),
    )

    df = df.with_columns(
        (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist"),
    )

    # Momentum and ROC
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(cfg.momentum_period)).alias("momentum"),
        ((pl.col("close") / pl.col("close").shift(cfg.roc_period) - 1) * 100).alias("roc"),
    )

    # RSI slope (momentum of RSI)
    df = df.with_columns(
        (pl.col("rsi") - pl.col("rsi").shift(3)).alias("rsi_slope"),
        (pl.col("macd_hist") - pl.col("macd_hist").shift(1)).alias("macd_hist_slope"),
    )

    return df


def _add_volatility_features(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    if "atr" not in df.columns:
        tr_hl = pl.col("high") - pl.col("low")
        tr_hc = (pl.col("high") - pl.col("close").shift(1)).abs()
        tr_lc = (pl.col("low") - pl.col("close").shift(1)).abs()

        df = df.with_columns(
            pl.max_horizontal(tr_hl, tr_hc, tr_lc).alias("true_range"),
        )

        df = df.with_columns(
            pl.col("true_range").rolling_mean(window_size=cfg.atr_period).alias("atr"),
        )

    # Bollinger Bands
    df = df.with_columns(
        pl.col("close").rolling_mean(window_size=cfg.bb_period).alias("bb_mid"),
        pl.col("close").rolling_std(window_size=cfg.bb_period).alias("bb_std_val"),
    )

    df = df.with_columns(
        (pl.col("bb_mid") + cfg.bb_std * pl.col("bb_std_val")).alias("bb_upper"),
        (pl.col("bb_mid") - cfg.bb_std * pl.col("bb_std_val")).alias("bb_lower"),
    )

    df = df.with_columns(
        ((pl.col("close") - pl.col("bb_lower")) / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-8)).alias(
            "bb_position"
        ),
        (pl.col("bb_upper") - pl.col("bb_lower")).alias("bb_width"),
        ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_mid")).alias("bb_width_pct"),
    )

    # ATR ratio (current volatility vs average)
    df = df.with_columns(
        (pl.col("atr") / pl.col("atr").rolling_mean(window_size=cfg.atr_period)).alias("atr_ratio"),
        (pl.col("atr") / pl.col("close") * 100).alias("atr_pct"),
    )

    df = df.drop(["bb_std_val"])

    return df


def _add_volume_features(df: pl.DataFrame, cfg: FeatureConfig) -> pl.DataFrame:
    if "volume_ma" not in df.columns:
        df = df.with_columns(
            pl.col("volume").rolling_mean(window_size=cfg.volume_ma_period).alias("volume_ma"),
        )

    df = df.with_columns(
        (pl.col("volume") / (pl.col("volume_ma") + 1e-8)).alias("volume_ratio"),
        pl.col("volume").rolling_std(window_size=cfg.volume_ma_period).alias("volume_std"),
    )

    df = df.with_columns(
        ((pl.col("volume") - pl.col("volume_ma")) / (pl.col("volume_std") + 1e-8)).alias("volume_zscore"),
    )

    # Volume trend
    df = df.with_columns(
        (pl.col("volume") / pl.col("volume").shift(1)).alias("volume_change"),
        (
            pl.col("volume").rolling_mean(window_size=5)
            / pl.col("volume").rolling_mean(window_size=20)
        ).alias("volume_trend"),
    )

    # On-Balance Volume momentum
    obv_sign = pl.when(pl.col("close") > pl.col("close").shift(1)).then(1).when(
        pl.col("close") < pl.col("close").shift(1)
    ).then(-1).otherwise(0)

    df = df.with_columns(
        (obv_sign * pl.col("volume")).cum_sum().alias("obv"),
    )

    df = df.with_columns(
        (pl.col("obv") - pl.col("obv").shift(5)).alias("obv_momentum"),
    )

    return df


def _add_pattern_features(df: pl.DataFrame) -> pl.DataFrame:
    body = pl.col("close") - pl.col("open")
    body_abs = body.abs()
    upper_shadow = pl.col("high") - pl.max_horizontal("open", "close")
    lower_shadow = pl.min_horizontal("open", "close") - pl.col("low")
    candle_range = pl.col("high") - pl.col("low")

    df = df.with_columns(
        (body_abs / (candle_range + 1e-8)).alias("body_pct"),
        (lower_shadow / (candle_range + 1e-8)).alias("lower_shadow_pct"),
        (upper_shadow / (candle_range + 1e-8)).alias("upper_shadow_pct"),
    )

    # Hammer score (higher = more hammer-like)
    df = df.with_columns(
        (
            pl.col("lower_shadow_pct") * 2
            - pl.col("body_pct")
            - pl.col("upper_shadow_pct")
        ).alias("hammer_score"),
    )

    # Consecutive down bars
    df = df.with_columns(
        (pl.col("close") < pl.col("open")).cast(pl.Int32).alias("is_red"),
    )

    df = df.with_columns(
        pl.col("is_red").rolling_sum(window_size=5).alias("red_count_5"),
    )

    # Price distance from recent high/low
    df = df.with_columns(
        ((pl.col("close") - pl.col("low").rolling_min(window_size=20)) /
         (pl.col("high").rolling_max(window_size=20) - pl.col("low").rolling_min(window_size=20) + 1e-8)).alias(
            "range_position_20"
        ),
    )

    df = df.drop(["is_red"])

    return df


def create_labels(
    df: pl.DataFrame,
    forward_window: int = 10,
    profit_threshold: float = 2.0,
    loss_threshold: float = 1.5,
) -> pl.DataFrame:
    df = df.with_columns(
        pl.col("close").shift(-forward_window).alias("future_close"),
        pl.col("high").rolling_max(window_size=forward_window).shift(-forward_window).alias("future_high"),
        pl.col("low").rolling_min(window_size=forward_window).shift(-forward_window).alias("future_low"),
    )

    df = df.with_columns(
        ((pl.col("future_high") - pl.col("close")) / pl.col("close") * 100).alias("max_gain_pct"),
        ((pl.col("close") - pl.col("future_low")) / pl.col("close") * 100).alias("max_loss_pct"),
        ((pl.col("future_close") - pl.col("close")) / pl.col("close") * 100).alias("return_pct"),
    )

    # Label: 1 = profitable trade (hit TP before SL), 0 = losing trade
    df = df.with_columns(
        (
            (pl.col("max_gain_pct") >= profit_threshold)
            & (pl.col("max_loss_pct") < loss_threshold)
        ).cast(pl.Int32).alias("label"),
    )

    return df


def get_feature_columns() -> list[str]:
    return [
        "wma_5", "wma_10", "wma_20", "wma_50",
        "price_wma20_ratio", "price_wma50_ratio",
        "hl_position", "body_ratio",
        "rsi", "rsi_slope",
        "macd", "macd_signal", "macd_hist", "macd_hist_slope",
        "momentum", "roc",
        "atr", "atr_ratio", "atr_pct",
        "bb_position", "bb_width_pct",
        "volume_ratio", "volume_zscore", "volume_trend",
        "obv_momentum",
        "body_pct", "lower_shadow_pct", "upper_shadow_pct", "hammer_score",
        "red_count_5", "range_position_20",
        "drop_pct", "recovery_pct",
    ]
