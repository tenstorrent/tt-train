#!/bin/bash

export TT_METAL_LOGGER_LEVEL=FATAL
SCRIPT="/home/ubuntu/ML-Framework-CPP/build/sources/examples/nano_gpt/nano_gpt"
RESET_BOARD="tt-smi -r 0"
SEED=5489

$RESET_BOARD
$SCRIPT -p transformer.msgpack -s $SEED -e 