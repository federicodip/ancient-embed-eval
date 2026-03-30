#!/usr/bin/env python3
"""Within-language retrieval evaluation.

Tests whether nearby chunks in embedding space belong to the same work/author.
Uses existing passage embeddings as queries — no model loading needed, runs fast.

For each sampled chunk, searches the FAISS index (excluding itself) and checks
if the nearest neighbors are from the same work or author.

Usage:
    python eval_within_lang.py --model bge-m3
    python eval_within_lang.py --model all
    python eval_within_lang.py --model bge-m3 --n-queries 2000
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss

from utils import load_config, get_model_config, load_embeddings, compute_mrr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def eval_within_lang_model(model_cfg, config, args):
    """Run within-language retrieval evaluation for a single model."""
    name = model_cfg["name"]
    embeddings_dir = Path(config["embeddings_dir"]) / name
    results_dir = Path(config["output_dir"]) / name
    results_dir.mkdir(parents=True, exist_ok=True)

    k_values = config["eval"]["k_values"]
    max_k = max(k_values)

    # Load embeddings
    log.info("Loading embeddings from %s", embeddings_dir)
    vectors, metadata = load_embeddings(embeddings_dir)
    log.info("Loaded %d vectors (dim=%d)", vectors.shape[0], vectors.shape[1])

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    log.info("FAISS index built (%d vectors)", index.ntotal)

    # Group chunks by language
    lang_indices = defaultdict(list)
    for i, m in enumerate(metadata):
        lang_indices[m.get("language", "unknown")].append(i)

    # Sample query chunks per language
    rng = np.random.default_rng(42)
    n_per_lang = args.n_queries

    results_by_lang = {}

    for lang, indices in sorted(lang_indices.items()):
        if len(indices) < 100:
            log.warning("Skipping language %s — only %d chunks", lang, len(indices))
            continue

        sample_idx = rng.choice(indices, min(n_per_lang, len(indices)), replace=False)
        log.info("Evaluating %d query chunks for [%s] (%d total chunks)", len(sample_idx), lang, len(indices))

        work_ranks = []  # rank of first same-work hit
        author_ranks = []  # rank of first same-author hit

        for qi in sample_idx:
            query_vec = vectors[qi:qi+1]
            q_meta = metadata[qi]
            q_work = q_meta.get("work_id", "")
            q_author = q_meta.get("author", "")

            # Search (fetch max_k + 1 since the chunk itself will be rank 1)
            scores, result_indices = index.search(query_vec, max_k + 1)
            result_indices = result_indices[0]

            work_rank = 0
            author_rank = 0
            rank = 0

            for ri in result_indices:
                if ri == qi:
                    continue  # skip self
                rank += 1
                if rank > max_k:
                    break

                r_meta = metadata[ri]
                if r_meta.get("work_id", "") == q_work and work_rank == 0:
                    work_rank = rank
                if r_meta.get("author", "") == q_author and author_rank == 0:
                    author_rank = rank

                if work_rank > 0 and author_rank > 0:
                    break

            work_ranks.append(work_rank)
            author_ranks.append(author_rank)

        # Compute metrics
        n = len(sample_idx)
        lang_metrics = {"n": n, "total_chunks": len(indices)}

        for k in k_values:
            work_hits = sum(1 for r in work_ranks if 0 < r <= k)
            author_hits = sum(1 for r in author_ranks if 0 < r <= k)
            lang_metrics[f"work_recall_at_{k}"] = round(work_hits / n, 4)
            lang_metrics[f"author_recall_at_{k}"] = round(author_hits / n, 4)

        lang_metrics["work_mrr"] = round(compute_mrr(work_ranks), 4)
        lang_metrics["author_mrr"] = round(compute_mrr(author_ranks), 4)

        results_by_lang[lang] = lang_metrics

    # Assemble and save
    result = {
        "model": name,
        "corpus_size": int(vectors.shape[0]),
        "embedding_dim": int(vectors.shape[1]),
        "n_queries_per_lang": n_per_lang,
        "by_language": results_by_lang,
    }

    out_path = results_dir / "within_lang_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("Saved within-language metrics to %s", out_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Within-Language Retrieval — {name}")
    print(f"{'='*60}")
    for lang, lm in sorted(results_by_lang.items()):
        print(f"\n  [{lang.upper()}] n={lm['n']} queries, {lm['total_chunks']} chunks")
        print(f"    Same-work  Recall@1={lm.get('work_recall_at_1', 0):.3f}  "
              f"R@5={lm.get('work_recall_at_5', 0):.3f}  "
              f"R@10={lm.get('work_recall_at_10', 0):.3f}  "
              f"MRR={lm['work_mrr']:.3f}")
        print(f"    Same-author Recall@1={lm.get('author_recall_at_1', 0):.3f}  "
              f"R@5={lm.get('author_recall_at_5', 0):.3f}  "
              f"R@10={lm.get('author_recall_at_10', 0):.3f}  "
              f"MRR={lm['author_mrr']:.3f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Within-language retrieval evaluation")
    parser.add_argument("--model", required=True, help="Model name or 'all'")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n-queries", type=int, default=1000, help="Number of query chunks per language")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model == "all":
        for model_cfg in config["models"]:
            embeddings_dir = Path(config["embeddings_dir"]) / model_cfg["name"]
            if not (embeddings_dir / "vectors.npy").exists():
                log.warning("Skipping %s — no embeddings found", model_cfg["name"])
                continue
            eval_within_lang_model(model_cfg, config, args)
    else:
        model_cfg = get_model_config(config, args.model)
        eval_within_lang_model(model_cfg, config, args)


if __name__ == "__main__":
    main()
