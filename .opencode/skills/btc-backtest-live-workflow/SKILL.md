---
name: btc-backtest-live-workflow
description: "Use when tasks involve BTC strategy research workflow, including baseline->parametric->ML/DL comparison, latest-1y test split, metrics reporting, pages publishing, and Go live-trading handoff with strategy notes/checkpoints. Trigger keywords: backtest, parametric, ML, DL, pages, hive partition, /mnt/data/finance, Go live trading, Svelte dashboard/history/settings."
---

# BTC Backtest to Live Workflow

## Purpose

Standardize how this repository executes V-Rebounder research and hands off to live trading.

## V-Rebound Strategy

The V-Rebounder system detects and trades V-shaped rebounds in Bitcoin:

1. **Capitulation Detection**: Sharp price drop with volume spike
2. **Bottom Identification**: Price stabilization at support
3. **Recovery Confirmation**: Rapid price recovery from bottom
4. **Entry Signal**: Long position when V-pattern confirmed

### Key Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| `drop_threshold_pct` | Minimum drop % for capitulation | 2-5% |
| `drop_window_bars` | Bars to measure drop | 3-10 |
| `recovery_threshold_pct` | Minimum recovery % from bottom | 1-3% |
| `recovery_window_bars` | Bars to confirm recovery | 2-5 |
| `volume_spike_mult` | Volume multiplier confirmation | 1.5-3.0x |
| `atr_multiplier` | ATR-based dynamic threshold | 1.5-3.0 |

## Inputs

- Strategy hypothesis or change request
- Data under `/mnt/data/finance` in hive partition layout
- Target timeframe(s): `5m`, `15m`, `1h`, `4h`

## Workflow

1. Baseline algorithm backtest (V-rebound with fixed thresholds)
2. Parametric study on key knobs (drop/recovery thresholds, volume, ATR)
3. ML/DL enhancement with strict time split
   - Train: all history except latest 1 year
   - Test: latest 1 year
4. Compare three tracks in a single summary
5. If profitable candidate exists, run ideation for hardening:
   - Risk controls
   - Execution realism (fees/slippage/latency)
   - Robustness and regime sensitivity

## Performance and Engine Rules

- Use Polars by default.
- Use DuckDB only with explicit need (join-heavy/SQL-heavy pipeline or IO benefit).
- Use GPU and parallel processing aggressively where feasible.

## Required Metrics

- Return
- Alpha
- Sharpe
- Max DD
- Win Rate
- Profit Factor
- Trades
- Expected return per trade

## Documentation Outputs

- Publish baseline strategy and result pages in `pages/` for GitHub Pages.
- Keep `docs/` aligned if legacy pages still consume docs artifacts.
- For live-trading-bound work, completion requires:
  - Strategy note
  - Checkpoint list with validation evidence

## Live Trading Baseline (Svelte)

- Required tabs: Dashboard, History, Settings
- Settings minimum fields: exchange `api_key`, `secret_key`, trading on/off toggle
