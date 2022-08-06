#!/bin/bash
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=32:mem=100gb

# Cluster Environment Setup
cd $PBS_O_WORKDIR

module load anaconda3
module load anaconda3/personal
source activate fenicsproject2

# Check that FIRST_PORT and NUM_PORT variables are set
# TODO
FIRST_PORT=8000
NUM_PORT=30
# check that all ports are free
output=$(python3 -c "from utils import bash_check_avail; bash_check_avail($FIRST_PORT, $NUM_PORT)")

if [ $output == "T" ]; then
    echo "Ports available, launch..."
else
    if [ $output == "F" ]; then
        echo "Abort; some ports are not avail"
        exit 0
    else
        echo "wrong output checking ports; abort"
        exit 1
    fi
fi

# if I went so far, all ports are free: can launch!

# launch everything:

# launch servers
echo "Launching the servers. This takes a few seconds..."
let "n_sec_sleep = 20 * $NUM_PORT"
echo "Wait $n_sec_sleep secs for servers to start..."

sleep 2

python3 launch_servers.py -p $FIRST_PORT -n $NUM_PORT&

sleep $n_sec_sleep

python3 launch_parallel_training.py -p $FIRST_PORT -n $NUM_PORT

echo "Launched training!"

exit 0
