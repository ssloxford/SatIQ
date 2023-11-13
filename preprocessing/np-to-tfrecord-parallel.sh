#!/bin/bash
# This script converts the numpy arrays to tfrecords in parallel, using a specified number of processes.

# If not enough arguments are specified, print help and exit
if [ $# -ne 4 ]; then
    echo "Usage: $0 <num_processes> <files_per_process> <path_in> <path_out>"
    exit 1
fi

# Take arguments from the command line
num_processes=$1
files_per_process=$2
path_in=$3
path_out=$4

num_files=$(ls -1 $path_in/samples_*.npy | wc -l)
step_size=$((num_processes*files_per_process))

# Run the conversion in parallel
for i in $(seq 0 $step_size $((num_files-1))); do
    start=$i
    end=$((start+step_size))
    if [ $end -gt $num_files ]; then
        end=$num_files
    fi
    echo "Starting processes with files $start to $end"
    for j in $(seq 0 $(($num_processes - 1))); do
        skip_files=$((start+j*files_per_process))
        echo "    Starting process $j with files $skip_files to $(($skip_files+files_per_process))"
        python3 np-to-tfrecord.py --path-in=$path_in --path-out=$path_out --skip-files=$skip_files --max-files=$files_per_process --no-shuffle &
    done
    wait
done