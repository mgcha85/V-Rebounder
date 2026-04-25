from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


@dataclass
class DataConfig:
    data_root: Path = Path("/mnt/data/finance/cryptocurrency")
    symbol: str = "BTCUSDT"


def load_btc_data(
    config: DataConfig | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pl.DataFrame:
    if config is None:
        config = DataConfig()

    data_path = config.data_root / config.symbol

    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    df = pl.scan_parquet(data_path / "**/*.parquet")

    if start_date:
        df = df.filter(pl.col("datetime") >= pl.lit(start_date).str.to_datetime())
    if end_date:
        df = df.filter(pl.col("datetime") <= pl.lit(end_date).str.to_datetime())

    result = df.collect().sort("datetime")

    if "open_time" not in result.columns or result["open_time"].dtype != pl.Datetime:
        result = result.with_columns(pl.col("datetime").alias("open_time"))

    return result


def resample_ohlcv(df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    timeframe_map = {
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
    }

    if timeframe not in timeframe_map:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Use one of {list(timeframe_map)}")

    return (
        df.sort("open_time")
        .group_by_dynamic("open_time", every=timeframe)
        .agg(
            pl.col("open").first().alias("open"),
            pl.col("high").max().alias("high"),
            pl.col("low").min().alias("low"),
            pl.col("close").last().alias("close"),
            pl.col("volume").sum().alias("volume"),
            pl.col("quote_volume").sum().alias("quote_volume"),
            pl.col("trades").sum().alias("trades"),
        )
    )


def add_indicators(df: pl.DataFrame, atr_period: int = 14, rsi_period: int = 14) -> pl.DataFrame:
    df = df.with_columns(
        # True Range components
        (pl.col("high") - pl.col("low")).alias("tr_hl"),
        (pl.col("high") - pl.col("close").shift(1)).abs().alias("tr_hc"),
        (pl.col("low") - pl.col("close").shift(1)).abs().alias("tr_lc"),
    )

    df = df.with_columns(
        pl.max_horizontal("tr_hl", "tr_hc", "tr_lc").alias("true_range"),
    )

    df = df.with_columns(
        pl.col("true_range").rolling_mean(window_size=atr_period).alias("atr"),
        pl.col("volume").rolling_mean(window_size=20).alias("volume_ma"),
        pl.col("volume").rolling_std(window_size=20).alias("volume_std"),
    )

    # RSI calculation
    df = df.with_columns(
        (pl.col("close") - pl.col("close").shift(1)).alias("price_change"),
    )

    df = df.with_columns(
        pl.when(pl.col("price_change") > 0)
        .then(pl.col("price_change"))
        .otherwise(0)
        .alias("gain"),
        pl.when(pl.col("price_change") < 0)
        .then(pl.col("price_change").abs())
        .otherwise(0)
        .alias("loss"),
    )

    df = df.with_columns(
        pl.col("gain").rolling_mean(window_size=rsi_period).alias("avg_gain"),
        pl.col("loss").rolling_mean(window_size=rsi_period).alias("avg_loss"),
    )

    df = df.with_columns(
        (100 - (100 / (1 + pl.col("avg_gain") / pl.col("avg_loss")))).alias("rsi"),
    )

    # Bollinger Bands
    df = df.with_columns(
        pl.col("close").rolling_mean(window_size=20).alias("bb_mid"),
        pl.col("close").rolling_std(window_size=20).alias("bb_std"),
    )

    df = df.with_columns(
        (pl.col("bb_mid") + 2 * pl.col("bb_std")).alias("bb_upper"),
        (pl.col("bb_mid") - 2 * pl.col("bb_std")).alias("bb_lower"),
    )

    return df.drop(["tr_hl", "tr_hc", "tr_lc", "price_change", "gain", "loss", "avg_gain", "avg_loss", "bb_std"])
