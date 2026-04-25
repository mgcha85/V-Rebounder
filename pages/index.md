# V-Rebounder Strategy Results

## Overview

V-shaped rebound detection strategy for Bitcoin trading. Detects sharp drops followed by rapid recovery and enters long positions.

**Data Period**: 2019-12-31 ~ 2026-04-12 (3.3M 1-minute bars)

**Test Period**: 2023-01-01 ~ 2026-04-12

---

## Strategy Track Comparison

### 1. Baseline Algorithm

Simple threshold-based detection:
- Drop ≥ 3%, Recovery ≥ 1.5%, Volume spike ≥ 2x

| Metric | 5m | 15m | 1h |
|--------|-----|------|-----|
| Return | -73.8% | N/A | N/A |
| Max DD | 73.8% | N/A | N/A |
| Win Rate | 36.0% | N/A | N/A |
| Trades | 164 | N/A | N/A |

**Conclusion**: Default parameters underperform significantly.

---

### 2. Parametric Study (5m Timeframe)

Grid search over 81 parameter combinations on 2023-2024 data.

**Top 5 Configurations (by Calmar Ratio)**:

| Rank | Drop | Recovery | TP | SL | Return | Max DD | Win Rate | Trades | Calmar |
|------|------|----------|----|----|--------|--------|----------|--------|--------|
| 1 | 4.0% | 1.5% | 2.0% | 1.5% | 9.5% | 17.9% | 50.0% | 32 | 0.53 |
| 2 | 5.0% | 1.5% | 4.0% | 1.0% | 5.0% | 9.6% | 40.0% | 20 | 0.52 |
| 3 | 4.0% | 1.5% | 4.0% | 1.5% | 8.3% | 16.3% | 33.3% | 30 | 0.51 |
| 4 | 5.0% | 1.0% | 4.0% | 1.0% | 5.8% | 11.6% | 39.1% | 23 | 0.50 |
| 5 | 5.0% | 1.0% | 4.0% | 2.0% | 7.6% | 15.6% | 45.5% | 22 | 0.49 |

---

### 3. Improved Detector (Research-Based)

Enhanced detection with:
- RSI favorable condition (oversold OR recovering)
- Volume exhaustion pattern (spike followed by decline)
- Dynamic ATR-based thresholds

**Timeframe Comparison**:

| Timeframe | Legacy Calmar | Improved Calmar | Improvement | Return | Max DD | Win Rate |
|-----------|---------------|-----------------|-------------|--------|--------|----------|
| **5m** | 0.53 | **1.36** | +157% | 16.9% | 12.4% | 53.3% |
| 15m | -0.51 | 0.20 | +139% | 5.9% | 30.1% | 49.2% |
| 1h | -0.86 | -0.87 | - | -53.3% | 61.5% | 35.1% |

**Best Configuration (5m-balanced)**:
- `drop_threshold_pct`: 4.0%
- `recovery_threshold_pct`: 1.5%
- `take_profit_pct`: 2.0%
- `stop_loss_pct`: 1.5%
- `volume_spike_mult`: 1.5x

---

### 4. ML/DL Enhancement

XGBoost classifier trained on 33 technical features.

**Features Used**:
- Price: WMA (5,10,20,50), price/WMA ratios
- Momentum: RSI, MACD, ROC, momentum
- Volatility: ATR, Bollinger Bands position/width
- Volume: volume ratio, z-score, OBV momentum
- Pattern: hammer score, body ratio, shadow percentages

**Training Results** (944 signals, 20% positive rate):

| Split | Accuracy | Precision | Recall | AUC |
|-------|----------|-----------|--------|-----|
| Train | 0.798 | 0.500 | 0.903 | - |
| Test (OOS) | 0.815 | 0.333 | 0.250 | **0.826** |

**Top 5 Feature Importance**:

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | atr_pct | 0.093 |
| 2 | bb_position | 0.039 |
| 3 | bb_width_pct | 0.039 |
| 4 | volume_zscore | 0.038 |
| 5 | rsi | 0.038 |

**ML Improvement**: +66.5% precision over baseline

---

## Final Metrics (Best Strategy: 5m Improved)

| Metric | Value |
|--------|-------|
| Total Return | **16.9%** |
| Max Drawdown | **12.4%** |
| Calmar Ratio | **1.36** |
| Win Rate | 53.3% |
| Total Trades | 30 |
| Expected Return/Trade | 0.56% |

---

## Practical Trading Simulation

### Assumptions

| Parameter | Value |
|-----------|-------|
| **Seed Capital** | $10,000 |
| **Position Size** | 20% per trade |
| **Leverage** | 10x |
| **Execution** | Market order at close |
| **Maker Fee** | 0.02% |
| **Taker Fee** | 0.05% |
| **Slippage** | 0.05% |
| **Funding Rate** | 0.01% per 8h |

### Risk Management

- Stop Loss: 1.5% (hard stop)
- Take Profit: 2.0%
- Half Close: 1.5% (50% position)
- Max Concurrent Positions: 1

---

## Reproduction

```bash
# Setup
cd V-Rebounder
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run baseline backtest
python scripts/backtest_5m.py

# Run parametric study
python scripts/parametric_study.py
```

---

## Conclusions

1. **5m timeframe performs best** - Higher frequency captures V-rebounds better
2. **Improved detector outperforms legacy** - Calmar 0.53 → 1.36 (+157%)
3. **ML provides marginal improvement** - AUC 0.826, but limited by data scarcity
4. **Key features**: ATR%, BB position, Volume z-score
5. **Longer timeframes (1h) not suitable** - Negative returns

---

*Generated: 2026-04-25*
*Data Source: Binance BTCUSDT Perpetual Futures*
