#!/bin/bash

export TT_METAL_LOGGER_LEVEL=FATAL
SCRIPT="/home/ubuntu/ML-Framework-CPP/build/sources/examples/nano_gpt/nano_gpt"
INTERVAL=100
DEFAULT_SEED=5489
MAX_STEPS=5000
SLEEP_DURATION=30

echo "Running $SCRIPT..."
for i in {1..10}; do
    $SCRIPT -i $INTERVAL -p transformer.msgpack -s $((DEFAULT_SEED - i)) -m $MAX_STEPS
    steps_done=$((i * MAX_STEPS)) 
    echo "Done $steps_done iterations" 
    sleep $SLEEP_DURATION
done
echo "Done running $SCRIPT"
