#!/bin/bash

# Start MATLAB and run task
cat << MATM | /usr/local/bin/matlab2020b -nosplash -nodisplay -singleCompThread
% Setup environment
run('../setup_env.m')

% Run task
bfeq_sweep_task($SLURM_ARRAY_TASK_ID);
MATM
