#!/usr/bin/bash -l
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=lowprio
#SBATCH --chdir=/home/fdipas/ancient-embed-eval
#SBATCH --output=logs/build-%j.out
#SBATCH --error=logs/build-%j.err
set -e

# Build the Apptainer container on a compute node.
# The cluster auto-binds /home, /scratch, etc. which breaks builds,
# so we clear APPTAINER_BINDPATH.

module load apptainer

mkdir -p /scratch/fdipas/ancient-embed-eval

HTTPS_PROXY=http://10.129.62.115:3128 \
HTTP_PROXY=http://10.129.62.115:3128 \
APPTAINER_BINDPATH="" \
    apptainer build /scratch/fdipas/ancient-embed-eval/container.sif container.def

echo "Container built: /scratch/fdipas/ancient-embed-eval/container.sif"
