#!/bin/sh

cd ~/Documents/fassion_mnist/

module load conda
export TMPDIR=$HOME/tmp
source activate conda_env

export LANGUAGE=en_US.UTF-8
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8
#locale-gen en_US.UTF-8

# ssh -L 8080:wn58:8080 wn58
# ~/Documents/fassion_mnist/tensorboard.sh

tensorboard --port 8080 --logdir ./tasks/apr_16_exp10_model_11_steam_room_cifar_10/runs/apr_16_exp10_model_11_steam_room_cifar_10_7307_7803