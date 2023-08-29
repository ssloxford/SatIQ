#!/bin/sh

#cd /
#echo $(cat /proc/sys/kernel/core_pattern)
#ulimit -c unlimited
SAMPLE_RATE=$((1000000*$MULTIPLIER))
#BURST_SAMPLE_RATE=2000000
BURST_SAMPLE_RATE=$((1000000*$MULTIPLIER))
NUM_SAMPLES=$((440*$MULTIPLIER))
BURST_POST_LEN=$((16000*$MULTIPLIER))
BURST_PRE_LEN=$((2048*$MULTIPLIER))
BURST_WIDTH=$((20000*$MULTIPLIER))
MAX_BURST_LEN=$((90000*$MULTIPLIER))
python3 /code/iridium_extractor/iridium_extractor.py --zmq-address-iq $ZMQ_ADDRESS_IQ --zmq-address-bytes $ZMQ_ADDRESS_BYTES --sample-rate $SAMPLE_RATE --burst-sample-rate $BURST_SAMPLE_RATE --num-samples $NUM_SAMPLES --burst-post-len $BURST_POST_LEN --burst-pre-len $BURST_PRE_LEN --burst-width $BURST_WIDTH --max-burst-len $MAX_BURST_LEN "$@" --parallelism=$PARALLELISM --gain=$GAIN --burst-threshold=$BURST_THRESHOLD
#echo a
#echo $(find /tmp | grep core)
