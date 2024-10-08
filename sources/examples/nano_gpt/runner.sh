#!/bin/bash

export TT_METAL_LOGGER_LEVEL=FATAL
SCRIPT="/home/ubuntu/ML-Framework-CPP/build/sources/examples/nano_gpt/nano_gpt"
RESET_BOARD="tt-smi -r 0"
INTERVAL=100
DEFAULT_SEED=5489
MAX_STEPS=5000
SLEEP_DURATION=30
BATCH_SIZE=8

$RESET_BOARD
echo "Running $SCRIPT..."
for i in {1..5}; do
    $SCRIPT -i $INTERVAL -p transformer_gpt2.msgpack -s $((DEFAULT_SEED - i)) -m $MAX_STEPS -b $BATCH_SIZE
    $RESET_BOARD
    echo "Sleeping for $SLEEP_DURATION seconds and restarting training..."
    sleep $SLEEP_DURATION
done
echo "Done running $SCRIPT"
