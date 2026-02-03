#!/bin/bash
set -e

# Set parameters
TICKER_SET="ETFS"
RISK_METRIC="mdd" # Options: "std", "mdd", "sharpe"
START_DATE="2010-01-01"
N_GEN=20
POP_SIZE=50
TRAIN_WINDOW=$((365*3))
REBALANCE_FREQ=90
SEED=42

USE_SMOOTHING="--use-smoothing" # Enable price smoothing
FILL_MISSING="--fill-missing"     # Enable missing day interpolation
NO_FILL_WEEKENDS="" # Set to "--no-fill-weekends" to disable weekend interpolation when --fill-missing is enabled
TRANS_COST="0.0025"               # Transaction cost (0.25%)
MIN_LIQ="500000"                  # Min Daily Turnover (PLN) to be investable
QUIET=""                          # Set to "--quiet" for less output
CALLBACK_INTERVAL=1              # Interval for callback logging
GIF_DURATION="0.5"                # Duration of each frame in evolution GIFs (seconds)

echo "========================================================"
echo "Starting Portfolio Optimization Pipeline"
echo "Strategy: NSGA-II Walk-Forward Optimization"
echo "Tickers: $TICKER_SET | Metric: $RISK_METRIC"
echo "Window: $TRAIN_WINDOW days | Rebalance: $REBALANCE_FREQ days"
echo "Cost: $TRANS_COST | Min Liq: $MIN_LIQ | Missing Fill: ${FILL_MISSING:-Disabled} | Weekend Fill: ${NO_FILL_WEEKENDS:-Enabled}"
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
    --callback-interval "$CALLBACK_INTERVAL" \
    --gif-duration "$GIF_DURATION" \
    $USE_SMOOTHING \
    $FILL_MISSING \
    $NO_FILL_WEEKENDS \
    $QUIET

echo "========================================================"
echo "Pipeline Finished!"
echo "Check the 'plots/' directory for results."
echo "========================================================"