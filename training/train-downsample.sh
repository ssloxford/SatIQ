#!/bin/bash

NUM_EPOCHS=200
STOP_AFTER=10

for i in 1 2 3 4 5 6 7 8
do
    python3 ae-triplet-conv-dataset-slices.py --save-dir /data/models/downsample --num-epochs $NUM_EPOCHS --model-name downsample-1-$i --num-days 16 --stop-after $STOP_AFTER --downsample 1 --save-loss
done
