#!/usr/bin/env python3
"""Evaluate clustering quality of embeddings by author and work.

Measures whether embeddings naturally group chunks by author/work
using k-means clustering and information-theoretic metrics.

Usage:
    python eval_clustering.py --model bge-m3
    python eval_clustering.py --model all --sample 10000
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_config, get_model_config, load_embeddings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def eval_clustering_model(model_cfg, config, args):
    """Run clustering evaluation for a single model."""
    name = model_cfg["name"]
    embeddings_dir = Path(config["embeddings_dir"]) / name
    results_dir = Path(config["output_dir"]) / name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    log.info("Loading embeddings from %s", embeddings_dir)
    vectors, metadata = load_embeddings(embeddings_dir)
    log.info("Loaded %d vectors (dim=%d)", vectors.shape[0], vectors.shape[1])

    # Subsample if requested (clustering 179k points is slow)
    if args.sample and args.sample < len(vectors):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(vectors), args.sample, replace=False)
        indices.sort()
        vectors = vectors[indices]
        metadata = [metadata[i] for i in indices]
        log.info("Subsampled to %d vectors", len(vectors))

    # Extract labels
    authors = [m["author"] for m in metadata]
    work_ids = [m.get("work_id", m.get("title", "")) for m in metadata]

    unique_authors = sorted(set(authors))
    unique_works = sorted(set(work_ids))
    log.info("Unique authors: %d, unique works: %d", len(unique_authors), len(unique_works))

    # Author-level clustering
    log.info("Running k-means for author clustering (k=%d)", len(unique_authors))
    author_labels_true = [unique_authors.index(a) for a in authors]
    km_author = MiniBatchKMeans(
        n_clusters=len(unique_authors),
        random_state=42,
        batch_size=1024,
        n_init=3,
    )
    author_labels_pred = km_author.fit_predict(vectors)

    author_ari = adjusted_rand_score(author_labels_true, author_labels_pred)
    author_nmi = normalized_mutual_info_score(author_labels_true, author_labels_pred)
    author_vm = v_measure_score(author_labels_true, author_labels_pred)
    log.info("Author clustering — ARI: %.4f, NMI: %.4f, V-measure: %.4f", author_ari, author_nmi, author_vm)

    # Work-level clustering
    log.info("Running k-means for work clustering (k=%d)", len(unique_works))
    work_labels_true = [unique_works.index(w) for w in work_ids]
    km_work = MiniBatchKMeans(
        n_clusters=min(len(unique_works), len(vectors)),
        random_state=42,
        batch_size=1024,
        n_init=3,
    )
    work_labels_pred = km_work.fit_predict(vectors)

    work_ari = adjusted_rand_score(work_labels_true, work_labels_pred)
    work_nmi = normalized_mutual_info_score(work_labels_true, work_labels_pred)
    work_vm = v_measure_score(work_labels_true, work_labels_pred)
    log.info("Work clustering — ARI: %.4f, NMI: %.4f, V-measure: %.4f", work_ari, work_nmi, work_vm)

    # Intra-author vs inter-author cosine similarity
    # Sample pairs to keep this tractable
    log.info("Computing intra/inter-author cosine similarity (sampled pairs)")
    rng = np.random.default_rng(42)
    n_pairs = min(5000, len(vectors))

    # Build author-to-indices map
    author_to_idx = {}
    for i, a in enumerate(authors):
        author_to_idx.setdefault(a, []).append(i)

    intra_sims = []
    inter_sims = []
    sampled = rng.choice(len(vectors), n_pairs, replace=False)

    for i in sampled:
        # Intra: pick another chunk from same author
        same = author_to_idx[authors[i]]
        if len(same) > 1:
            j = rng.choice(same)
            while j == i and len(same) > 1:
                j = rng.choice(same)
            sim = cosine_similarity(vectors[i:i+1], vectors[j:j+1])[0][0]
            intra_sims.append(float(sim))

        # Inter: pick a chunk from a different author
        diff_author = authors[i]
        while diff_author == authors[i]:
            k = rng.integers(len(vectors))
            diff_author = authors[k]
        sim = cosine_similarity(vectors[i:i+1], vectors[k:k+1])[0][0]
        inter_sims.append(float(sim))

    intra_mean = np.mean(intra_sims) if intra_sims else 0.0
    inter_mean = np.mean(inter_sims) if inter_sims else 0.0
    separation = intra_mean - inter_mean
    log.info("Intra-author sim: %.4f, Inter-author sim: %.4f, Gap: %.4f",
             intra_mean, inter_mean, separation)

    # Assemble results
    result = {
        "model": name,
        "corpus_size": len(metadata),
        "embedding_dim": int(vectors.shape[1]),
        "sample_size": args.sample,
        "num_authors": len(unique_authors),
        "num_works": len(unique_works),
        "author_clustering": {
            "ari": round(author_ari, 4),
            "nmi": round(author_nmi, 4),
            "v_measure": round(author_vm, 4),
        },
        "work_clustering": {
            "ari": round(work_ari, 4),
            "nmi": round(work_nmi, 4),
            "v_measure": round(work_vm, 4),
        },
        "similarity": {
            "intra_author_mean": round(intra_mean, 4),
            "inter_author_mean": round(inter_mean, 4),
            "separation_gap": round(separation, 4),
            "n_pairs_sampled": n_pairs,
        },
    }

    with open(results_dir / "clustering_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log.info("Saved clustering metrics to %s", results_dir / "clustering_metrics.json")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Clustering — {name}")
    print(f"{'='*60}")
    print(f"  Author ARI:       {author_ari:.4f}")
    print(f"  Author NMI:       {author_nmi:.4f}")
    print(f"  Author V-measure: {author_vm:.4f}")
    print(f"  Work ARI:         {work_ari:.4f}")
    print(f"  Work NMI:         {work_nmi:.4f}")
    print(f"  Work V-measure:   {work_vm:.4f}")
    print(f"  Intra-author sim: {intra_mean:.4f}")
    print(f"  Inter-author sim: {inter_mean:.4f}")
    print(f"  Separation gap:   {separation:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Evaluate embedding clustering quality")
    parser.add_argument("--model", required=True, help="Model name or 'all'")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--sample", type=int, default=None, help="Subsample N chunks for speed")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model == "all":
        for model_cfg in config["models"]:
            embeddings_dir = Path(config["embeddings_dir"]) / model_cfg["name"]
            if not (embeddings_dir / "vectors.npy").exists():
                log.warning("Skipping %s — no embeddings found", model_cfg["name"])
                continue
            eval_clustering_model(model_cfg, config, args)
    else:
        model_cfg = get_model_config(config, args.model)
        eval_clustering_model(model_cfg, config, args)


if __name__ == "__main__":
    main()
