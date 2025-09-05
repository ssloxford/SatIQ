#!/bin/bash

NUM_EPOCHS=200
NUM_DAYS=16
STOP_AFTER=10

FILTER_PROPERTY=noise

for i in -90 -92 -94 -96 -98 -100 -102 -104
do
    python3 ae-triplet-conv-dataset-slices.py --num-epochs $NUM_EPOCHS --model-name days-$NUM_DAYS-$FILTER_PROPERTY-lt-$i --num-days $NUM_DAYS --stop-after $STOP_AFTER --filter-property $FILTER_PROPERTY --filter-value $i
done
