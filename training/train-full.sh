#!/bin/bash

NUM_EPOCHS=200
STOP_AFTER=10

for i in -1 1 2 4 8 16 32 64
do
    python3 ae-triplet-conv-dataset-slices.py --num-epochs $NUM_EPOCHS --model-name days-$i --num-days $i --stop-after $STOP_AFTER
done
