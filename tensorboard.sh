#!/bin/sh

cd ~/Documents/fassion_minst/

module load conda
export TMPDIR=$HOME/tmp
source activate conda_env

export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
#locale-gen en_US.UTF-8

# ssh -L 8080:wn58:8080 wn58
# ~/Documents/fassion_minst/tensorboard.sh

tensorboard --port 8080 --samples_per_plugin images=0,scalars=0,audio=0,projector=0 \
--logdir /mnt/home/evaldsu/Documents/fassion_minst/tasks/oct_28_model_13_hospital_exp13_vggface_full_rep_radam_fixed_colored/runs/oct_28_model_13_hospital_exp13_vggface_full_rep_radam_fixed_colored_22502_24846