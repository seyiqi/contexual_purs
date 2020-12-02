#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-22:00:00
#SBATCH --mem=10GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=beer_baseline
#SBATCH --mail-type=END

module purge
module load python3/intel/3.6.3
module load tensorflow/python3.6/1.3.0

cd /home/nw1045/courses/contexual_purs

source env.sh

python3 code/train_purs.py \
--device /gpu:0 \
--batch-size 32 \
--dataset beer \
--learning-rate $lr$ \
--save-path $savepath$

