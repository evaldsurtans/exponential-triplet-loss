#!/bin/sh -v
#PBS -e /mnt/home/evaldsu/Documents/fassion_mnist/tasks/process_data_vggface2
#PBS -o /mnt/home/evaldsu/Documents/fassion_mnist/tasks/process_data_vggface2
#PBS -q batch
#PBS -p 1000
#PBS -l nodes=1:ppn=24
#PBS -l mem=100gb
#PBS -l walltime=96:00:00

module load conda
eval "$(conda shell.bash hook)"
source activate conda_env
mkdir /scratch/evalds
mkdir /scratch/evalds/tmp
mkdir /scratch/evalds/data
export TMPDIR=/scratch/evalds/tmp
export TEMP=/scratch/evalds/tmp
export SDL_AUDIODRIVER=waveout
export SDL_VIDEODRIVER=x11

cd ~/Documents/fassion_mnist/

python ./process_data_vggface2.py -path_input /mnt/home/evaldsu/data_raw/vggface2/test -path_output /mnt/home/evaldsu/data_raw/vggface2

