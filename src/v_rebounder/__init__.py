"""V-Rebounder: V-shaped rebound detection and trading system for Bitcoin."""

__version__ = "0.1.0"

from .detector import VReboundDetector
from .strategy import VReboundStrategy
from .data_loader import load_btc_data, resample_ohlcv

__all__ = [
    "VReboundDetector",
    "VReboundStrategy",
    "load_btc_data",
    "resample_ohlcv",
]
