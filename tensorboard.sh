#!/bin/sh

cd ~/Documents/fassion_minst/

module load conda
export TMPDIR=$HOME/tmp
source activate conda_env

export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
#locale-gen en_US.UTF-8

# ssh -L 8080:wn57:8080 wn57
# ~/Documents/fassion_minst/tensorboard.sh

tensorboard --port 8080 --logdir ./tasks/test_17_exp_pair/runs