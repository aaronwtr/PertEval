#!/bin/bash
#$ -cwd
#$ -m bea
#$ -N perturbench-train-scgpt
#$ -j y
#$ -o output_train-scgpt.log
#$ -pe smp 1          # 8 cores per GPU
#$ -l h_rt=1:0:0      # 1 hours runtime for shortq
#$ -l h_vmem=11G      # 11G RAM per core

export OMP_NUM_THREADS=1
export HYDRA_FULL_ERROR=1 
module load python cudnn
source venv/bin/activate

python src/train.py experiment=mlp_norman_train #--multirun hydra=spectra_data_sweep paths=cluster
