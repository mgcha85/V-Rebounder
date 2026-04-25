# PLAN.md

## Task

- Request:
- Owner:
- Date:

## Goal

- Primary outcome:
- Non-goals:

## V-Rebound Strategy Focus

Core detection parameters to optimize:

- `drop_threshold_pct`: Minimum drop % for capitulation (2-5%)
- `drop_window_bars`: Bars to measure drop (3-10)
- `recovery_threshold_pct`: Minimum recovery % from bottom (1-3%)
- `recovery_window_bars`: Bars to confirm recovery (2-5)
- `volume_spike_mult`: Volume multiplier for confirmation (1.5-3.0x)
- `atr_multiplier`: ATR-based dynamic threshold (1.5-3.0)

## Strategy Track (Required Order)

1. Baseline algorithm (V-rebound detection with fixed thresholds)
2. Parametric study (grid search on V-rebound parameters)
3. ML/DL (pattern classification/filtering enhancement)

Comparison note:

- [ ] All three tracks compared in one summary table
- [ ] Profitable candidate selected for ideation (if any)

## Scope

- In scope files/modules:
- Out of scope:

## Data and Infra Assumptions

- Data root: `/mnt/data/finance`
- Time-series format: hive partition
- Compute plan:
   - [ ] GPU usage considered
   - [ ] Parallel processing plan defined
- Engine plan:
   - [ ] Polars default
   - [ ] DuckDB usage rationale documented (if used)

## Milestones

1. Context scan
   - [ ] Read affected modules
   - [ ] Confirm data/paths and dependencies
2. Design
   - [ ] Define minimal change set
   - [ ] Define verification strategy
3. Implementation
   - [ ] Apply edits
   - [ ] Update docs/artifacts if needed
4. Verification
   - [ ] Run lint/tests/scripts as applicable
   - [ ] Record known gaps
5. Reporting
   - [ ] Required metrics table completed
   - [ ] Pages content update prepared
6. Handoff (for live trading scope)
   - [ ] Strategy note drafted
   - [ ] Checkpoint evidence attached
   - [ ] F/E build and B/E serving integration plan defined
   - [ ] Telegram notification plan for completed trades (`open -> close`) defined
   - [ ] Settings toggle behavior (`on/off`) for Telegram notifications defined

## Verification Gates

- Gate A (static):
- Gate B (runtime):
- Gate C (artifact consistency):

## Metrics Contract

- [ ] Return
- [ ] Alpha
- [ ] Sharpe
- [ ] Max DD
- [ ] Win Rate
- [ ] Profit Factor
- [ ] Trades
- [ ] Expected return per trade

## ML/DL Protocol

- [ ] Train/Test split is time-based
- [ ] Test window is the most recent 1 year
- [ ] Train window is all earlier history

## Risks and Mitigations

- Risk:
  - Mitigation:

## Expected Deliverables

- Code:
- Documents:
- Generated artifacts:

## Publishing Targets

- `pages/` updated for GitHub Pages
- `docs/` sync note (if legacy pages still consume docs)
- `pages/` includes practical trading simulation section with:
   - Seed capital
   - Position size per trade
   - Execution model and cost assumptions (fees/slippage/leverage)
