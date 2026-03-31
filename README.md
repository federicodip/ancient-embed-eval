# Ancient Embedding Evaluation Pipeline

Benchmarks embedding models on Ancient Greek and Latin text retrieval. Measures how well each model retrieves the correct author's passages given English-language queries about classical texts — without any metadata filtering, testing raw embedding quality.

## Data

- **Corpus:** 179,301 chunks from Perseus Digital Library (95k Greek, 84k Latin)
- **Queries:** 171 English questions with gold author/work labels

## Models tested

All open-weights, all multilingual.

| Name | HuggingFace ID | Params | Dim | License |
|------|---------------|--------|-----|---------|
| bge-m3 | BAAI/bge-m3 | 568M | 1024 | MIT |
| multilingual-e5-large | intfloat/multilingual-e5-large | 560M | 1024 | MIT |
| sphilberta | bowphs/SPhilBerta | 100M | 768 | Apache 2.0 |
| qwen3-embed-8B | Qwen/Qwen3-Embedding-8B | 8B | 4096 | Apache 2.0 |
| nemotron-embed-8B | nvidia/llama-embed-nemotron-8b | 8B | 4096 | NSCL v1 |

SPhilBERTa is the only embedding model purpose-built for Ancient Greek and Latin (Heidelberg NLP). The others are general-purpose multilingual models.

## Results

### Cross-lingual retrieval (English queries → Ancient Greek/Latin passages)

| Model | Params | R@1 | R@5 | R@10 | R@50 | MRR |
|-------|--------|-----|-----|------|------|-----|
| **qwen3-embed-8B** | 8B | **0.345** | **0.655** | **0.766** | **0.977** | **0.499** |
| sphilberta | 100M | 0.210 | 0.462 | 0.579 | 0.772 | 0.326 |
| bge-m3 | 568M | 0.170 | 0.363 | 0.462 | 0.731 | 0.265 |
| nemotron-embed-8B | 8B | 0.152 | 0.304 | 0.386 | 0.684 | 0.228 |
| multilingual-e5-large | 560M | 0.117 | 0.269 | 0.363 | 0.719 | 0.200 |

### Per-language breakdown (Author Recall@10 / MRR)

| Model | Full corpus | Latin-only | Greek-only |
|-------|:-----------:|:----------:|:----------:|
| **qwen3-embed-8B** | **0.766 / 0.499** | **0.857 / 0.673** | 0.804 / 0.475 |
| nemotron-embed-8B | 0.386 / 0.228 | 0.667 / 0.395 | **0.804 / 0.550** |
| sphilberta | 0.579 / 0.326 | 0.587 / 0.403 | 0.705 / 0.410 |
| bge-m3 | 0.462 / 0.265 | 0.778 / 0.570 | 0.455 / 0.219 |
| multilingual-e5-large | 0.363 / 0.200 | 0.682 / 0.431 | 0.420 / 0.180 |

### Within-language retrieval (chunk → same-work neighbors, R@10 / MRR)

| Model | Greek | Latin |
|-------|-------|-------|
| **bge-m3** | **0.888 / 0.754** | **0.891 / 0.701** |
| multilingual-e5-large | 0.832 / 0.645 | 0.897 / 0.708 |
| qwen3-embed-8B | 0.815 / 0.620 | 0.850 / 0.647 |
| nemotron-embed-8B | 0.794 / 0.587 | 0.803 / 0.572 |
| sphilberta | 0.714 / 0.476 | 0.621 / 0.388 |

### Key findings

1. **Qwen3-8B dominates cross-lingual retrieval** — nearly double the Recall@10 of baselines (77% vs 46%)
2. **SPhilBERTa (100M) beats models 5-80x its size** — outperforms BGE-M3 (568M) and Nemotron (8B) on the full corpus despite being purpose-built with far fewer parameters
3. **SPhilBERTa excels on Greek** (70.5% R@10), approaching the 8B models (80.4%), but is weaker on Latin (58.7%) where general-purpose models have more training signal
4. **Latin is easier than Greek** for all general-purpose models, but SPhilBERTa reverses this pattern — its Classical Philology training gives it a Greek advantage
5. **Nemotron-8B wins on Greek-only** corpus (highest MRR 0.55), despite underperforming on mixed corpus
6. **Smaller encoder models win within-language** — BGE-M3 produces tighter same-work clusters than 8B models or SPhilBERTa
7. **Model choice depends on use case**: cross-lingual RAG → Qwen3-8B; Greek without GPU → SPhilBERTa; same-language clustering → BGE-M3

## Pipeline

```
embed_corpus.py    →  embeddings/{model}/vectors.npy          (GPU, ~30-60 min per model)
eval_retrieval.py  →  results/{model}/retrieval_metrics.json   (GPU, ~2 min per model)
eval_within_lang.py → results/{model}/within_lang_metrics.json (CPU, ~2 min per model)
eval_clustering.py →  results/{model}/clustering_metrics.json  (CPU, ~2 min per model)
compare.py         →  comparison tables, CSV, and plots        (CPU, instant)
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

# Build container
sbatch jobs/build_container.sh
```

### Run embeddings

```bash
# Smaller models (~20-30 min each on GPU)
sbatch jobs/embed.sh bge-m3
sbatch jobs/embed.sh multilingual-e5-large

# 8B models (~1 hour each, need 80GB GPU)
sbatch jobs/embed.sh qwen3-embed-8B
sbatch jobs/embed.sh nemotron-embed-8B
```

### Run evaluation

```bash
sbatch jobs/eval.sh all              # full corpus + clustering + within-language
sbatch jobs/eval.sh all latin        # Latin-only
sbatch jobs/eval.sh all greek        # Greek-only
```

### View results

```bash
module load apptainer
apptainer exec --nv /scratch/fdipas/ancient-embed-eval/container.sif python compare.py --plot --output results/comparison.csv
```

## Metrics

- **Author Recall@k** — fraction of queries with the correct author in top-k results
- **Work Recall@k** — correct author AND correct work in top-k (stricter)
- **MRR** — Mean Reciprocal Rank of first correct hit
- **NDCG@k** — Normalized Discounted Cumulative Gain
- **ARI / NMI** — clustering agreement between embeddings and true author/work labels
- **Within-language retrieval** — given a chunk, can the model find other chunks from the same work? Tests source-language understanding without cross-lingual transfer

## Project structure

```
config.yaml            # model definitions, paths, eval parameters
utils.py               # data loading, author matching, metrics
embed_corpus.py        # embed chunks → vectors.npy
eval_retrieval.py      # FAISS search → retrieval metrics
eval_within_lang.py    # within-language retrieval (chunk → same-work neighbors)
eval_clustering.py     # k-means → clustering metrics
compare.py             # aggregate comparison tables + plots
container.def          # Apptainer definition (CUDA + PyTorch)
jobs/
  build_container.sh   # Slurm: build container
  embed.sh             # Slurm: embed with GPU
  eval.sh              # Slurm: run eval (GPU)
data/
  chunks/              # corpus JSONL (gitignored, 558MB)
  queries/             # query set JSONL
embeddings/            # saved vectors (gitignored)
results/               # output metrics and plots
```
