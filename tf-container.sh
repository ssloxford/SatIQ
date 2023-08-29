#!/bin/bash

docker build -t sat-iq .
docker run --gpus all -it --rm -u $(id -u):$(id -g) -v $(pwd):/code -v $(pwd)/../data:/data sat-iq
