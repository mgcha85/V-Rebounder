# V-Rebounder

**V-shaped Rebound Detection and Automated Trading System for Bitcoin**

## Overview

V-Rebounder는 비트코인의 급락 후 빠른 반등(V자 회복) 패턴을 탐지하고 자동으로 매매하는 시스템입니다.

### 핵심 전략

비트코인 시장에서 자주 발생하는 패턴:
1. **급락 (Capitulation)**: 패닉 셀링으로 인한 급격한 가격 하락
2. **꼬리 형성 (Tail/Wick)**: 하락 후 빠르게 매수세 유입
3. **V자 반등 (V-Rebound)**: 급격한 가격 회복

이 패턴을 실시간으로 탐지하여 롱 포지션 진입 타이밍을 포착합니다.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        V-Rebounder                               │
├─────────────────────────────────────────────────────────────────┤
│  Research (Python)          │  Live Trading (Go + Svelte)       │
│  ─────────────────          │  ─────────────────────────        │
│  • Backtest Engine          │  • Real-time Detection            │
│  • Parametric Study         │  • Position Management            │
│  • ML/DL Enhancement        │  • Binance Futures API            │
│  • Strategy Optimization    │  • Web Dashboard                  │
└─────────────────────────────────────────────────────────────────┘
```

## V-Rebound Detection Logic

### 기본 파라미터

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `drop_threshold_pct` | 급락 인정 최소 하락률 | 2-5% |
| `drop_window_bars` | 급락 측정 기간 (bars) | 3-10 |
| `recovery_threshold_pct` | 회복 인정 최소 상승률 | 1-3% |
| `recovery_window_bars` | 회복 확인 기간 (bars) | 2-5 |
| `volume_spike_mult` | 거래량 스파이크 배수 | 1.5-3.0x |
| `atr_multiplier` | ATR 기반 동적 임계값 | 1.5-3.0 |

### 진입/청산 조건

**진입 (Entry)**:
1. 최근 N bars 내 `drop_threshold_pct` 이상 하락 감지
2. 거래량 스파이크 확인 (capitulation signal)
3. 저점에서 `recovery_threshold_pct` 이상 회복 시작
4. RSI oversold 조건 (optional)

**청산 (Exit)**:
- Take Profit: `take_profit_pct` 도달
- Stop Loss: `stop_loss_pct` 도달
- Half Close: `half_close_pct` 도달 시 절반 청산

## Project Structure

```
V-Rebounder/
├── src/v_rebounder/           # Python backtest package
│   ├── detector.py            # V-rebound pattern detection
│   ├── strategy.py            # Entry/exit logic, backtest engine
│   ├── parametric.py          # Parameter grid study
│   ├── data_loader.py         # Data loading and resampling
│   ├── features.py            # Feature engineering for ML
│   ├── models_tree.py         # Tree-based models (XGBoost, LightGBM)
│   └── models_deep.py         # Deep learning models (LSTM, Transformer)
├── scripts/                   # Executable research scripts
│   ├── backtest_5m.py
│   ├── backtest_15m.py
│   └── parametric_study.py
├── live-trading/
│   ├── engine/                # Go trading engine
│   │   ├── cmd/main.go
│   │   └── internal/
│   │       ├── config/        # Strategy profiles
│   │       ├── detector/      # V-rebound detection
│   │       ├── engine/        # Trading logic
│   │       ├── exchange/      # Binance API
│   │       └── server/        # HTTP API
│   └── frontend/              # Svelte dashboard
│       └── src/
│           ├── routes/        # Pages (Dashboard, History, Settings)
│           └── lib/           # API client, config
├── pages/                     # GitHub Pages (results)
└── docs/                      # Strategy notes, checkpoints
```

## Research Workflow

순차적 전략 검증:

1. **Baseline Algorithm** - 기본 V-rebound 탐지 로직
2. **Parametric Study** - 파라미터 그리드 서치 최적화
3. **ML/DL Enhancement** - 머신러닝 기반 필터링/예측

### ML/DL 데이터 분할

- **Train**: 전체 기간 - 최근 1년
- **Test**: 최근 1년 (out-of-sample)

## Required Metrics

모든 백테스트 결과에 포함:

- Return (%)
- Alpha
- Sharpe Ratio
- Max Drawdown (%)
- Win Rate (%)
- Profit Factor
- Total Trades
- Expected Return per Trade (%)

## Data

- **Root**: `/mnt/data/finance`
- **Format**: Hive partition
- **Timeframes**: 5m, 15m, 1h, 4h

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backtest | Python (Polars, pandas-ta) |
| ML/DL | PyTorch, XGBoost, LightGBM |
| Live Engine | Go |
| Frontend | Svelte |
| Exchange | Binance Futures |
| Data | Polars, DuckDB |

## Quick Start

```bash
# Setup Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Run backtest
python scripts/backtest_5m.py

# Run parametric study
python scripts/parametric_study.py

# Start live trading (testnet)
cd live-trading/engine
TESTNET=true go run cmd/main.go

# Start frontend
cd live-trading/frontend
npm run dev
```

## Live Trading Checklist

프로덕션 배포 전 필수 확인:

- [ ] Backtest strategy note 작성 완료
- [ ] Testnet에서 1주일 이상 검증
- [ ] Risk management 파라미터 설정
- [ ] Telegram notification 설정
- [ ] API key 보안 확인

## License

Private - Internal Use Only
