#!/bin/bash
#SBATCH --job-name=colbert_testing
#SBATCH --output=slurm_output_test-colbert.txt
#SBATCH --gres=gpu:1
#SBATCH --mem=64GB





# Directory where you keep the beir-ColBERT repo
cd /local_directory/beir-ColBERT
source ~/.bashrc
# Set your conda virtual environment
conda activate conda_env_name




# Dataset name
dataset=dataset_name
NUM_PARTITIONS=32768
# Prepared target collection & test queries in the tsv format (please refer to beir-ColBERT for the data formatting: https://github.com/thakur-nandan/beir-ColBERT)
COLLECTION=local_path_to_target_collection/${dataset}_collections.tsv
QUERIES=local_path_to_test_queries/${dataset}_queries.tsv
INDEX_NAME=${dataset}_colbert
# Path to the trained colbert model (only needs the directory having *.dnn file)
CHECKPOINT=/local/scratch/rchan31/NAACL_Github/data/step8_trained_colbert_model/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-124.dnn
## if we want to automatically detect the *.dnn file inside the directory, then use the below line
# CHECKPOINT=$(find "local_path_to_root_directory/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/" -name "colbert-*.dnn" -type f)
echo "Loading the checkpoint file >>> $CHECKPOINT"

ROOT_DIR=path_to_save_logs_and_ranking_output
OUTPUT_DIR=${ROOT_DIR}/output
INDEX_ROOT=${ROOT_DIR}/index
RANKING_DIR=${ROOT_DIR}/ranking

#####################################################################################################################################
#                                                                 (1) Indexing
#####################################################################################################################################
python -m torch.distributed.run \
  --nproc_per_node=1 -m colbert.index \
  --root $OUTPUT_DIR \
  --doc_maxlen 300 \
  --mask-punctuation \
  --bsize 128 \
  --amp \
  --checkpoint $CHECKPOINT \
  --index_root $INDEX_ROOT \
  --index_name $INDEX_NAME \
  --collection $COLLECTION \
  --experiment ${dataset}

#####################################################################################################################################
#                                                                 (2) Faiss Indexing
#####################################################################################################################################
python -m colbert.index_faiss \
  --index_root $INDEX_ROOT \
  --index_name $INDEX_NAME \
  --partitions $NUM_PARTITIONS \
  --sample 0.3 \
  --root $OUTPUT_DIR \
  --experiment ${dataset}

#####################################################################################################################################
#                                                                 (3) ANN Search
#####################################################################################################################################
python -m colbert.retrieve \
  --amp \
  --doc_maxlen 300 \
  --mask-punctuation \
  --bsize 256 \
  --queries $QUERIES \
  --nprobe 32 \
  --partitions $NUM_PARTITIONS \
  --faiss_depth 100 \
  --depth 100 \
  --index_root $INDEX_ROOT \
  --index_name $INDEX_NAME \
  --checkpoint $CHECKPOINT \
  --root $OUTPUT_DIR \
  --experiment ${dataset} \
  --ranking_dir $RANKING_DIR

#####################################################################################################################################
#                                                                 (4) BEIR Evaluation
#####################################################################################################################################
python -m colbert.beir_eval \
  --dataset ${dataset} \
  --split "test" \
  --collection $COLLECTION \
  --rankings "${RANKING_DIR}/ranking.tsv"

echo "===================================================================================="
echo "              Completed testing with dataset: ${dataset}"
echo "===================================================================================="