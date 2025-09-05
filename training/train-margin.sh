#!/bin/bash

NUM_EPOCHS=200
STOP_AFTER=10
NUM_DAYS=16

for i in 0.5 0.1 0.05 0.01 2.0
do
    python3 ae-triplet-conv-dataset-slices.py --num-epochs $NUM_EPOCHS --model-name days-$NUM_DAYS-margin-$i --num-days $NUM_DAYS --stop-after $STOP_AFTER --triplet-margin $i
done
