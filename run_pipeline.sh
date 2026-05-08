#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <parameter.json>"
    exit 1
fi

PARAMS="$1"

echo "=== [1/3] Predicting ==="
python predict.py "$PARAMS"

echo "=== [2/3] Extracting ==="
python extract_daisy.py "$PARAMS"

echo "=== [3/3] Matching to roots ==="
python ./match_to_roots/get_roots_synapses.py -p "$PARAMS"

echo "=== Done ==="
