#!/usr/bin/bash -l
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/ancient-embed-eval
#SBATCH --output=logs/embed-%j.out
#SBATCH --error=logs/embed-%j.err
set -e

# Embed corpus chunks with a model.
# Usage: sbatch jobs/embed.sh bge-m3
#        sbatch jobs/embed.sh all

MODEL=${1:?Usage: sbatch jobs/embed.sh <model-name|all>}

module load apptainer

HF_HOME=/scratch/fdipas/cache/huggingface \
HTTPS_PROXY=http://10.129.62.115:3128 \
HTTP_PROXY=http://10.129.62.115:3128 \
    apptainer exec --nv /scratch/fdipas/ancient-embed-eval/container.sif \
    python embed_corpus.py --model "$MODEL" --resume

echo "Done: embed $MODEL"
