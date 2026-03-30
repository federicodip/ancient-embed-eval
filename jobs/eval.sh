#!/usr/bin/bash -l
#SBATCH --time=02:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/ancient-embed-eval
#SBATCH --output=logs/eval-%j.out
#SBATCH --error=logs/eval-%j.err
set -e

# Run retrieval + clustering + within-language eval.
# Uses GPU for query encoding (8B models are too slow on CPU).
# Usage: sbatch jobs/eval.sh bge-m3              # full eval
#        sbatch jobs/eval.sh all                  # all models
#        sbatch jobs/eval.sh bge-m3 latin         # Latin-only eval
#        sbatch jobs/eval.sh all greek             # all models, Greek-only

MODEL=${1:?Usage: sbatch jobs/eval.sh <model-name|all> [latin|greek]}
LANGUAGE=${2:-}

LANG_FLAG=""
if [ -n "$LANGUAGE" ]; then
    LANG_FLAG="--language $LANGUAGE"
fi

module load apptainer

HF_HOME=/scratch/fdipas/cache/huggingface \
HTTPS_PROXY=http://10.129.62.115:3128 \
HTTP_PROXY=http://10.129.62.115:3128 \
    apptainer exec --nv /scratch/fdipas/ancient-embed-eval/container.sif \
    python eval_retrieval.py --model "$MODEL" $LANG_FLAG

# Only run clustering and within-language eval on full corpus
if [ -z "$LANGUAGE" ]; then
    HF_HOME=/scratch/fdipas/cache/huggingface \
        apptainer exec --nv /scratch/fdipas/ancient-embed-eval/container.sif \
        python eval_clustering.py --model "$MODEL" --sample 10000

    apptainer exec --nv /scratch/fdipas/ancient-embed-eval/container.sif \
        python eval_within_lang.py --model "$MODEL" --n-queries 1000
fi

echo "Done: eval $MODEL $LANGUAGE"
