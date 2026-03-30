#!/usr/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/ancient-embed-eval
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
set -e

# Run retrieval + clustering eval for a model (CPU only, no GPU needed).
# Usage: sbatch jobs/eval.sh bge-m3
#        sbatch jobs/eval.sh all

MODEL=${1:?Usage: sbatch jobs/eval.sh <model-name|all>}

module load apptainer

HF_HOME=/scratch/fdipas/cache/huggingface \
HTTPS_PROXY=http://10.129.62.115:3128 \
HTTP_PROXY=http://10.129.62.115:3128 \
    apptainer exec /scratch/fdipas/ancient-embed-eval/container.sif \
    python eval_retrieval.py --model "$MODEL"

HF_HOME=/scratch/fdipas/cache/huggingface \
    apptainer exec /scratch/fdipas/ancient-embed-eval/container.sif \
    python eval_clustering.py --model "$MODEL" --sample 10000

echo "Done: eval $MODEL"
