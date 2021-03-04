#!/bin/bash

# Local variables
N_TASK=72
LOG_DIR='logs/bfeq_job'

# Variables needed by bfeq.sh
# N_PARALLEL=1 for single element
# needed to use one slot per 5GB of memory
export N_PARALLEL=6
export RESULT_DIR='results'

# Make directories if needed
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

# Submit job to grid
LLsub ./bfeq.sh -s $N_PARALLEL -c xeon-e5 -t 1-$N_TASK:1 -o $LOG_DIR/log

