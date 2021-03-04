#!/usr/bin/bash

#set the slots (s parameter) to be 1/5GB, so -s 7 for tasks that require 32GB.
#LLsub ./MatlabCMD.sh -s 4 -c xeon-e5 -t 1-1800 -o /tmp/AM20128/grid.out.$JOB_ID.$TASK_ID -- datadir

if [ "$#" -lt 1 ]; then
    dirname="$(date +%Y-%m-%dT%H%M%z)"
    mkdir -p "../data/$dirname/workspaces"
    mkdir -p "../data/$dirname/figs"
    gitout=$(git status --porcelain)
    if [ ! -z "$gitout" ]
    then
        echo "Error: uncommitted changes. Please commit all changes before running."
        exit 22
    fi

    githash=$(git rev-parse HEAD)
    echo $githash > ../data/$dirname/GITHASH
    echo $0 > ../data/$dirname/SWEEPSCRIPT

    echo "LLsub /bin/bash 154 -s 4 -c xeon-e5 -t 1-154 -o ~/tmp/grid.out.\$JOB_ID.\$TASK_ID -- ./hardware_sim_batch_script.sh ../data/$dirname/workspaces"
else
    # Execute MATLAB
    cat <<-MATM |  /usr/local/matlabr2017a/bin/matlab -nosplash -nodisplay -singleCompThread
%
% Execute the Matlab function
%
hardware_sim_sweep_task($SLURM_ARRAY_JOB_ID,$SLURM_ARRAY_TASK_ID, '$1');
MATM
fi
