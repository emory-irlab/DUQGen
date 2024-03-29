#!/bin/bash
#SBATCH --job-name=colbert_training
#SBATCH --output=slurm_output_train-colbert.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB




# Directory where you keep the beir-ColBERT repo
cd /local_directory/beir-ColBERT
source ~/.bashrc
# Set your conda virtual environment name
conda activate conda_env_name





# Input data file to train
train_filepath=path_to_input_training_data_file.tsv
# Directory to save the trained colbert model
trained_root=path_to_save_trained_colbert_model
# Path to the MS-MARCO pre-trained ColBERT checkpoint
checkpoint_path=path_to_colbert_pretraind_on_msmarco/colbert-300000.dnn




python -m torch.distributed.run --nproc_per_node=1 -m colbert.train \
            --amp --doc_maxlen 300 --mask-punctuation --bsize 32 --accum 1 \
            --triples ${train_filepath} \
            --root ${trained_root} --experiment MSMARCO-psg --similarity l2 --run_name msmarco.psg.l2 \
            --lr 3e-6 \
            --checkpoint $checkpoint_path