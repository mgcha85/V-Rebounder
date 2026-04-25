# IMPLEMENT.md

## Task Summary

- Task: V-Rebounder Python backtest implementation
- Date: 2026-04-25
- Agent: Sisyphus

## Data and Runtime

- Data root used: `/mnt/data/finance/cryptocurrency`
- Hive partition verified: Yes (BTCUSDT date partitioned)
- Data range: 2019-12-31 ~ 2026-04-12 (3.3M rows, 1m bars)
- Compute used (GPU/parallel): Polars vectorized operations
- Engine choice: Polars (default per AGENTS.md)

## V-Rebounder Strategy Overview

Core detection logic for V-shaped rebounds in Bitcoin:

1. **Capitulation Detection**: Sharp price drop with volume spike
2. **Bottom Formation**: Rolling low within drop window
3. **Recovery Confirmation**: Price recovers from bottom within recovery window
4. **Entry Signal**: Long position when drop + recovery + volume conditions met

## Changes Applied

- File: `pyproject.toml`
  - Why: Python project configuration
  - What changed: Dependencies (polars, pandas-ta, xgboost, etc.)

- File: `src/v_rebounder/__init__.py`
  - Why: Package initialization
  - What changed: Public API exports

- File: `src/v_rebounder/data_loader.py`
  - Why: Data loading from hive partitions
  - What changed: load_btc_data, resample_ohlcv, add_indicators (ATR, RSI, BB)

- File: `src/v_rebounder/detector.py`
  - Why: V-rebound pattern detection
  - What changed: VReboundDetector with configurable thresholds, candlestick patterns

- File: `src/v_rebounder/strategy.py`
  - Why: Backtest engine
  - What changed: VReboundStrategy with TP/SL/half-close, fee simulation

- File: `src/v_rebounder/parametric.py`
  - Why: Parameter optimization
  - What changed: Grid search with Calmar ratio scoring

- File: `scripts/backtest_5m.py`
  - Why: Baseline backtest script
  - What changed: 5m timeframe backtest with default params

- File: `scripts/parametric_study.py`
  - Why: Parameter optimization script
  - What changed: Full grid search across V-rebound params

## Strategy Track Results

### Baseline (Default Parameters)

| Metric | Value |
|--------|-------|
| Return | -73.77% |
| Max DD | 73.77% |
| Win Rate | 35.98% |
| Profit Factor | 0.68 |
| Trades | 164 |

**Note**: Default params (drop=3%, recovery=1.5%) underperform - optimization needed.

### Parametric Study (2023-2024 Data, 5m Timeframe)

**Top 3 Parameter Combinations (by Calmar Ratio)**:

| Rank | drop | recovery | TP | SL | Return | Max DD | Win Rate | Trades | Calmar |
|------|------|----------|----|----|--------|--------|----------|--------|--------|
| 1 | 4.0% | 1.5% | 2.0% | 1.5% | 9.5% | 17.9% | 50.0% | 32 | 0.53 |
| 2 | 5.0% | 1.5% | 4.0% | 1.0% | 5.0% | 9.6% | 40.0% | 20 | 0.52 |
| 3 | 4.0% | 1.5% | 4.0% | 1.5% | 8.3% | 16.3% | 33.3% | 30 | 0.51 |

### Improved Detector (Research-Based)

리서치 결과 반영:
- RSI favorable 조건 추가 (oversold + recovering)
- Volume exhaustion 패턴 (스파이크 후 감소)
- Dead cat bounce 필터 (옵션)

**Timeframe 비교 (2023~ Data, Improved Detector)**:

| Timeframe | Legacy Calmar | Improved Calmar | Return | Max DD | Win Rate |
|-----------|---------------|-----------------|--------|--------|----------|
| **5m** | 0.53 | **1.36** (+157%) | 16.9% | 12.4% | 53.3% |
| 15m | -0.51 | **0.20** | 5.9% | 30.1% | 49.2% |
| 1h | -0.86 | -0.87 | -53.3% | 61.5% | 35.1% |

**결론**: 5m 타임프레임에서 개선된 detector가 최적 (Calmar 0.53 → 1.36)

### ML/DL Enhancement

**구현 완료**:
- `features.py`: 33개 피처 (WMA, MACD, RSI, BB, ATR, Volume, Pattern)
- `models_tree.py`: XGBoost/LightGBM 분류기

**학습 결과** (Relaxed detector, 944 signals):
- Train: Acc 0.798, Precision 0.500, Recall 0.903
- Test (OOS 1Y): AUC **0.826**, Precision 0.333 (baseline 0.200 대비 +66.5%)

**Top 5 Features**:
1. `atr_pct` (0.093)
2. `bb_position` (0.039)
3. `bb_width_pct` (0.039)
4. `volume_zscore` (0.038)
5. `rsi` (0.038)

**한계점**: 테스트 기간(최근 1년)에 V-rebound 신호가 적어 (6개) ML 효과 검증 어려움

## Metrics Summary (Best Config - 5m Improved)

| Metric | Value |
|--------|-------|
| Return | 16.9% |
| Max DD | 12.4% |
| Win Rate | 53.3% |
| Trades | 30 |
| Calmar Ratio | **1.36** |

## Verification

- Commands run:
  - `python scripts/backtest_5m.py` - Success
  - Grid search (81 combinations, 2023-2024) - Completed
  - Improved detector test (5m/15m/1h) - Completed
  - ML training (XGBoost) - Completed

## Next Steps

1. ~~Run ML/DL enhancement~~ - Completed
2. ~~Test on 15m, 1h timeframes~~ - Completed
3. Implement live trading engine (Go)
4. Implement frontend (Svelte)
5. More data collection for ML validation
