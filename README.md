# Ancient Embedding Evaluation Pipeline

Benchmarks embedding models on Ancient Greek and Latin text retrieval. Measures how well each model retrieves the correct author's passages given English-language queries about classical texts — without any metadata filtering, testing raw embedding quality.

## Data

- **Corpus:** 179,301 chunks from Perseus Digital Library (95k Greek, 84k Latin)
- **Queries:** 171 English questions with gold author/work labels

## Models tested

All open-weights, all multilingual (100+ languages).

| Name | HuggingFace ID | Params | Dim | License |
|------|---------------|--------|-----|---------|
| bge-m3 | BAAI/bge-m3 | 568M | 1024 | MIT |
| multilingual-e5-large | intfloat/multilingual-e5-large | 560M | 1024 | MIT |
| gte-multilingual | Alibaba-NLP/gte-multilingual-base | 305M | 768 | Apache 2.0 |
| qwen3-embed-0.6B | Qwen/Qwen3-Embedding-0.6B | 0.6B | 32-1024 | Apache 2.0 |
| embedding-gemma-300M | google/embeddinggemma-300m | 300M | 128-768 | Gemma |
| qwen3-embed-8B | Qwen/Qwen3-Embedding-8B | 8B | 32-7168 | Apache 2.0 |
| nemotron-embed-8B | nvidia/llama-embed-nemotron-8b | 8B | 4096 | NSCL v1 |

The 8B models need 80GB+ GPU memory. The smaller models run on any GPU (or CPU for eval only).

## Pipeline

```
embed_corpus.py   →  embeddings/{model}/vectors.npy     (GPU, slow)
eval_retrieval.py →  results/{model}/retrieval_metrics.json   (CPU, fast)
eval_clustering.py → results/{model}/clustering_metrics.json  (CPU, moderate)
compare.py        →  comparison table + plots                 (CPU, instant)
```

## Quick start (local, small sample)

```bash
pip install -r requirements.txt
python embed_corpus.py --model bge-m3 --sample 1000
python eval_retrieval.py --model bge-m3
python compare.py
```

### Language-filtered evaluation

Evaluate on a single language (Latin or Greek) — filters both the corpus index and the queries:

```bash
python eval_retrieval.py --model bge-m3 --language latin    # ~84k Latin chunks only
python eval_retrieval.py --model bge-m3 --language greek    # ~95k Greek chunks only
```

Results go to `results/{model}/latin/` and `results/{model}/greek/` respectively.

## Cluster deployment (UZH ScienceCluster)

### One-time setup

```bash
ssh fdipas@cluster.s3it.uzh.ch

# Create directories
mkdir -p ~/ancient-embed-eval/logs
mkdir -p /scratch/fdipas/ancient-embed-eval

# Symlink data and embeddings to scratch
ln -s /scratch/fdipas/ancient-embed-eval/data ~/ancient-embed-eval/data
ln -s /scratch/fdipas/ancient-embed-eval/embeddings ~/ancient-embed-eval/embeddings

# Copy data to scratch
scp -r data/ fdipas@cluster.s3it.uzh.ch:/scratch/fdipas/ancient-embed-eval/data/

# Build container (includes flash-attn for 8B models)
sbatch jobs/build_container.sh
```

### Run embeddings

```bash
# Smaller models (~1-2 hours each on GPU)
sbatch jobs/embed.sh bge-m3
sbatch jobs/embed.sh multilingual-e5-large
sbatch jobs/embed.sh qwen3-embed-0.6B

# 8B models (~3-4 hours each, need 80GB GPU)
sbatch jobs/embed.sh qwen3-embed-8B
sbatch jobs/embed.sh nemotron-embed-8B

# Or all models sequentially
sbatch jobs/embed.sh all
```

### Run evaluation

```bash
sbatch jobs/eval.sh bge-m3
# or after all embeddings are done:
sbatch jobs/eval.sh all
```

### View results

```bash
module load apptainer
apptainer exec /scratch/fdipas/ancient-embed-eval/container.sif python compare.py
apptainer exec /scratch/fdipas/ancient-embed-eval/container.sif python compare.py --plot
```

## Metrics

- **Author Recall@k** — fraction of queries with the correct author in top-k results
- **Work Recall@k** — correct author AND correct work in top-k (stricter)
- **MRR** — Mean Reciprocal Rank of first correct hit
- **NDCG@k** — Normalized Discounted Cumulative Gain
- **ARI / NMI** — clustering agreement between embeddings and true author/work labels

## Project structure

```
config.yaml           # model definitions, paths, eval parameters
utils.py              # data loading, author matching, metrics
embed_corpus.py       # embed chunks → vectors.npy
eval_retrieval.py     # FAISS search → retrieval metrics
eval_clustering.py    # k-means → clustering metrics
compare.py            # aggregate comparison tables + plots
container.def         # Apptainer definition (CUDA + PyTorch + flash-attn)
jobs/
  build_container.sh  # Slurm: build container
  embed.sh            # Slurm: embed with GPU
  eval.sh             # Slurm: run eval (CPU)
data/
  chunks/             # corpus JSONL
  queries/            # query set JSONL
embeddings/           # saved vectors (gitignored)
results/              # output metrics and plots
```
