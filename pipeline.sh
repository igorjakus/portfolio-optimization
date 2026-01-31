#!/bin/bash
set -e

# Set parameters
TICKER_SET="WIG_BROAD"
RISK_METRIC="sharpe"
START_DATE="2010-01-01"
N_GEN=20
POP_SIZE=50
TRAIN_WINDOW=1008
REBALANCE_FREQ=90
SEED=42

USE_SMOOTHING="--use-smoothing" # Enable price smoothing
FILL_MISSING="--fill-missing"     # Enable missing day interpolation
TRANS_COST="0.0025"               # Transaction cost (0.25%)
MIN_LIQ="500000"                  # Min Daily Turnover (PLN) to be investable
QUIET=""                          # Set to "--quiet" for less output

echo "========================================================"
echo "Starting Portfolio Optimization Pipeline"
echo "Strategy: NSGA-II Walk-Forward Optimization"
echo "Tickers: $TICKER_SET | Metric: $RISK_METRIC"
echo "Window: $TRAIN_WINDOW days | Rebalance: $REBALANCE_FREQ days"
echo "Cost: $TRANS_COST | Min Liq: $MIN_LIQ | Missing Fill: ${FILL_MISSING:-Disabled}"
echo "Seed: $SEED"
echo "========================================================"

uv run main.py \
    --ticker-set "$TICKER_SET" \
    --risk-metric "$RISK_METRIC" \
    --start-date "$START_DATE" \
    --n-generations "$N_GEN" \
    --pop-size "$POP_SIZE" \
    --train-window "$TRAIN_WINDOW" \
    --rebalance-freq "$REBALANCE_FREQ" \
    --transaction-cost "$TRANS_COST" \
    --min-liquidity "$MIN_LIQ" \
    --seed "$SEED" \
    $USE_SMOOTHING \
    $FILL_MISSING \
    $QUIET

echo "========================================================"
echo "Pipeline Finished!"
echo "Check the 'plots/' directory for results."
echo "========================================================"