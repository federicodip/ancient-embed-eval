#!/usr/bin/env python3
"""Evaluate retrieval quality for a model's embeddings.

Loads pre-computed embeddings, builds a FAISS index, runs queries,
and computes Recall@k, MRR, NDCG@k with per-language and per-author breakdowns.

Usage:
    python eval_retrieval.py --model bge-m3
    python eval_retrieval.py --model all
    python eval_retrieval.py --model bge-m3 --k 20 --verbose
"""

import argparse
import json
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import faiss

from embed_corpus import load_st_model
from utils import (
    load_config, get_model_config, load_queries, load_embeddings,
    authors_match, works_match,
    compute_mrr, compute_recall_at_k, compute_ndcg_at_k,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def encode_queries(model, texts, model_cfg):
    """Encode query texts using the correct mode for this model.

    - standard:    model.encode(texts) with optional query_prefix prepended
    - prompt_name: model.encode(texts, prompt_name="query")
    - asymmetric:  model.encode_query(texts)
    """
    normalize = model_cfg.get("normalize", True)
    encode_mode = model_cfg.get("encode_mode", "standard")

    if encode_mode == "asymmetric":
        return model.encode_query(
            texts,
            normalize_embeddings=normalize,
        )
    elif encode_mode == "prompt_name":
        return model.encode(
            texts,
            prompt_name="query",
            normalize_embeddings=normalize,
        )
    else:
        # standard — prefix already prepended by caller
        return model.encode(
            texts,
            normalize_embeddings=normalize,
        )


def eval_model(model_cfg, config, args):
    """Run retrieval evaluation for a single model."""
    name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    encode_mode = model_cfg.get("encode_mode", "standard")
    query_prefix = model_cfg.get("query_prefix", "")

    embeddings_dir = Path(config["embeddings_dir"]) / name
    results_dir = Path(config["output_dir"]) / name
    results_dir.mkdir(parents=True, exist_ok=True)

    k_values = config["eval"]["k_values"]
    max_k = max(k_values)

    # Load embeddings
    log.info("Loading embeddings from %s", embeddings_dir)
    vectors, metadata = load_embeddings(embeddings_dir)
    log.info("Loaded %d vectors (dim=%d)", vectors.shape[0], vectors.shape[1])

    # Build FAISS index (flat, cosine = normalized dot product)
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors = cosine
    index.add(vectors)
    log.info("FAISS index built (%d vectors)", index.ntotal)

    # Load model for query embedding
    model = load_st_model(model_cfg)

    # Load queries
    queries = load_queries(config["queries"])
    log.info("Loaded %d queries", len(queries))

    # Build author→language lookup (so we don't scan metadata per query)
    author_lang = {}
    for meta in metadata:
        a = meta["author"]
        if a not in author_lang:
            author_lang[a] = meta.get("language", "unknown")

    # Run evaluation
    per_query_results = []
    author_ranks = []  # rank of first author-matching hit (0 = not found)
    work_ranks = []    # rank of first work-matching hit

    # Per-language tracking
    lang_results = defaultdict(lambda: {"author_ranks": [], "work_ranks": []})
    # Per-author tracking
    author_results = defaultdict(lambda: {"author_ranks": [], "work_ranks": [], "n": 0})

    for q in queries:
        query_text = q["question"]
        expected_author = q["author"]
        expected_work = q.get("work", "")

        # Prepare query text — for standard mode, prepend prefix
        if encode_mode == "standard" and query_prefix:
            query_text_enc = query_prefix + query_text
        else:
            query_text_enc = query_text

        # Encode query
        query_vec = encode_queries(model, [query_text_enc], model_cfg).astype(np.float32)

        # Search
        scores, indices = index.search(query_vec, max_k)
        scores = scores[0]
        indices = indices[0]

        # Find first author hit and first work hit
        author_rank = 0
        work_rank = 0
        top_results = []

        for rank_idx, (idx, score) in enumerate(zip(indices, scores), start=1):
            meta = metadata[idx]
            a_match = authors_match(expected_author, meta["author"])
            w_match = a_match and works_match(expected_work, meta["title"], meta.get("work_id", ""))

            if a_match and author_rank == 0:
                author_rank = rank_idx
            if w_match and work_rank == 0:
                work_rank = rank_idx

            if args.verbose or rank_idx <= 5:
                top_results.append({
                    "rank": rank_idx,
                    "score": round(float(score), 4),
                    "author": meta["author"],
                    "title": meta["title"],
                    "work_id": meta.get("work_id", ""),
                    "author_match": a_match,
                    "work_match": w_match,
                })

        author_ranks.append(author_rank)
        work_ranks.append(work_rank)

        # Determine query language from the expected author's corpus chunks
        query_lang = "unknown"
        for corpus_author, lang in author_lang.items():
            if authors_match(expected_author, corpus_author):
                query_lang = lang
                break

        lang_results[query_lang]["author_ranks"].append(author_rank)
        lang_results[query_lang]["work_ranks"].append(work_rank)
        author_results[expected_author]["author_ranks"].append(author_rank)
        author_results[expected_author]["work_ranks"].append(work_rank)
        author_results[expected_author]["n"] += 1

        per_query_results.append({
            "query_id": q.get("id", ""),
            "question": q["question"],
            "expected_author": expected_author,
            "expected_work": expected_work,
            "language": query_lang,
            "author_first_rank": author_rank,
            "work_first_rank": work_rank,
            "top_results": top_results,
        })

    # Compute aggregate metrics
    n = len(queries)

    metrics = {}
    for k in k_values:
        author_hits = [1 for r in author_ranks if 0 < r <= k]
        work_hits = [1 for r in work_ranks if 0 < r <= k]
        metrics[f"author_recall_at_{k}"] = round(len(author_hits) / n, 4)
        metrics[f"work_recall_at_{k}"] = round(len(work_hits) / n, 4)
        metrics[f"ndcg_at_{k}"] = round(compute_ndcg_at_k(author_ranks, k), 4)

    metrics["author_mrr"] = round(compute_mrr(author_ranks), 4)
    metrics["work_mrr"] = round(compute_mrr(work_ranks), 4)

    # Per-language breakdown
    by_language = {}
    for lang, data in sorted(lang_results.items()):
        n_lang = len(data["author_ranks"])
        lang_metrics = {"n": n_lang}
        for k in k_values:
            hits = [1 for r in data["author_ranks"] if 0 < r <= k]
            lang_metrics[f"author_recall_at_{k}"] = round(len(hits) / n_lang, 4) if n_lang else 0
        lang_metrics["author_mrr"] = round(compute_mrr(data["author_ranks"]), 4)
        by_language[lang] = lang_metrics

    # Per-author breakdown
    by_author = {}
    for author, data in sorted(author_results.items()):
        n_auth = data["n"]
        auth_metrics = {"n": n_auth}
        for k in [10, 20, 50]:
            hits = [1 for r in data["author_ranks"] if 0 < r <= k]
            auth_metrics[f"author_recall_at_{k}"] = round(len(hits) / n_auth, 4) if n_auth else 0
        auth_metrics["author_mrr"] = round(compute_mrr(data["author_ranks"]), 4)
        by_author[author] = auth_metrics

    # Assemble full results
    result = {
        "model": name,
        "hf_id": hf_id,
        "corpus_size": int(vectors.shape[0]),
        "num_queries": n,
        "embedding_dim": int(vectors.shape[1]),
        "metrics": metrics,
        "by_language": by_language,
        "by_author": by_author,
    }

    # Save
    with open(results_dir / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log.info("Saved metrics to %s", results_dir / "retrieval_metrics.json")

    with open(results_dir / "retrieval_details.jsonl", "w", encoding="utf-8") as f:
        for r in per_query_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved per-query details to %s", results_dir / "retrieval_details.jsonl")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Model: {name}")
    print(f"Corpus: {vectors.shape[0]} chunks, {vectors.shape[1]}-dim")
    print(f"Queries: {n}")
    print(f"{'='*60}")
    print(f"{'Metric':<25} {'Value':>8}")
    print(f"{'-'*33}")
    for k in k_values:
        print(f"  Author Recall@{k:<4}       {metrics[f'author_recall_at_{k}']:>8.4f}")
    print(f"  Author MRR              {metrics['author_mrr']:>8.4f}")
    for k in k_values:
        print(f"  Work Recall@{k:<4}         {metrics[f'work_recall_at_{k}']:>8.4f}")
    print(f"  Work MRR                {metrics['work_mrr']:>8.4f}")
    print()
    for lang, lm in by_language.items():
        print(f"  [{lang}] n={lm['n']}, Recall@10={lm.get('author_recall_at_10', 'N/A')}, MRR={lm['author_mrr']}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--model", required=True, help="Model name or 'all'")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--k", type=int, default=None, help="Override max k value")
    parser.add_argument("--verbose", action="store_true", help="Include all top-k results in details")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.k:
        config["eval"]["k_values"] = [k for k in config["eval"]["k_values"] if k <= args.k] + [args.k]
        config["eval"]["k_values"] = sorted(set(config["eval"]["k_values"]))

    if args.model == "all":
        for model_cfg in config["models"]:
            embeddings_dir = Path(config["embeddings_dir"]) / model_cfg["name"]
            if not (embeddings_dir / "vectors.npy").exists():
                log.warning("Skipping %s — no embeddings found at %s", model_cfg["name"], embeddings_dir)
                continue
            eval_model(model_cfg, config, args)
    else:
        model_cfg = get_model_config(config, args.model)
        eval_model(model_cfg, config, args)


if __name__ == "__main__":
    main()
