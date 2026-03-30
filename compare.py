#!/usr/bin/env python3
"""Aggregate results across all models into comparison tables and plots.

Usage:
    python compare.py                              # print table
    python compare.py --output results/comparison.csv
    python compare.py --plot                       # generate bar charts
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_model_results(results_dir, model_name):
    """Load retrieval and clustering metrics for a model. Returns dict or None."""
    model_dir = Path(results_dir) / model_name
    result = {"model": model_name}

    retrieval_path = model_dir / "retrieval_metrics.json"
    if retrieval_path.exists():
        with open(retrieval_path, encoding="utf-8") as f:
            result["retrieval"] = json.load(f)
    else:
        result["retrieval"] = None

    clustering_path = model_dir / "clustering_metrics.json"
    if clustering_path.exists():
        with open(clustering_path, encoding="utf-8") as f:
            result["clustering"] = json.load(f)
    else:
        result["clustering"] = None

    return result


def print_comparison_table(all_results, k_values):
    """Print a formatted comparison table to stdout."""
    # Header
    header_parts = [f"{'Model':<25}", f"{'Dim':>5}"]
    for k in k_values:
        header_parts.append(f"{'R@'+str(k):>7}")
    header_parts.extend([f"{'MRR':>7}", f"{'ARI':>7}", f"{'NMI':>7}"])
    header = " ".join(header_parts)

    print(f"\n{'='*len(header)}")
    print("RETRIEVAL COMPARISON (Author Recall)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))

    for r in all_results:
        parts = [f"{r['model']:<25}"]
        ret = r.get("retrieval")
        clust = r.get("clustering")

        dim = ret["embedding_dim"] if ret else "?"
        parts.append(f"{dim:>5}")

        for k in k_values:
            val = ret["metrics"].get(f"author_recall_at_{k}", 0) if ret else 0
            parts.append(f"{val:>7.3f}")

        mrr = ret["metrics"].get("author_mrr", 0) if ret else 0
        parts.append(f"{mrr:>7.3f}")

        ari = clust["author_clustering"]["ari"] if clust else 0
        nmi = clust["author_clustering"]["nmi"] if clust else 0
        parts.append(f"{ari:>7.3f}" if clust else f"{'—':>7}")
        parts.append(f"{nmi:>7.3f}" if clust else f"{'—':>7}")

        print(" ".join(parts))

    # Per-language breakdown
    print(f"\n{'='*80}")
    print("PER-LANGUAGE BREAKDOWN (Author Recall@10, MRR)")
    print(f"{'='*80}")

    languages = set()
    for r in all_results:
        if r.get("retrieval") and "by_language" in r["retrieval"]:
            languages.update(r["retrieval"]["by_language"].keys())

    for lang in sorted(languages):
        print(f"\n  [{lang.upper()}]")
        print(f"  {'Model':<25} {'n':>5} {'R@10':>7} {'MRR':>7}")
        print(f"  {'-'*47}")
        for r in all_results:
            ret = r.get("retrieval")
            if not ret or lang not in ret.get("by_language", {}):
                continue
            lm = ret["by_language"][lang]
            print(f"  {r['model']:<25} {lm['n']:>5} "
                  f"{lm.get('author_recall_at_10', 0):>7.3f} "
                  f"{lm.get('author_mrr', 0):>7.3f}")

    print()


def save_csv(all_results, k_values, output_path):
    """Save comparison table as CSV."""
    fieldnames = ["model", "dim"]
    for k in k_values:
        fieldnames.append(f"author_recall_at_{k}")
    fieldnames.extend(["author_mrr", "work_mrr", "author_ari", "author_nmi"])

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in all_results:
            ret = r.get("retrieval")
            clust = r.get("clustering")
            row = {"model": r["model"]}
            row["dim"] = ret["embedding_dim"] if ret else ""

            if ret:
                for k in k_values:
                    row[f"author_recall_at_{k}"] = ret["metrics"].get(f"author_recall_at_{k}", "")
                row["author_mrr"] = ret["metrics"].get("author_mrr", "")
                row["work_mrr"] = ret["metrics"].get("work_mrr", "")

            if clust:
                row["author_ari"] = clust["author_clustering"]["ari"]
                row["author_nmi"] = clust["author_clustering"]["nmi"]

            writer.writerow(row)

    log.info("Saved CSV to %s", output_path)


def generate_plots(all_results, k_values, output_dir):
    """Generate comparison bar charts."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [r["model"] for r in all_results]
    x = range(len(models))

    # Recall@k bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.8 / len(k_values)
    for i, k in enumerate(k_values):
        vals = []
        for r in all_results:
            ret = r.get("retrieval")
            vals.append(ret["metrics"].get(f"author_recall_at_{k}", 0) if ret else 0)
        offset = (i - len(k_values) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], vals, width, label=f"Recall@{k}")

    ax.set_ylabel("Author Recall")
    ax.set_title("Embedding Model Comparison — Author Recall@k")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_comparison.png", dpi=150)
    log.info("Saved recall plot to %s", output_dir / "recall_comparison.png")
    plt.close()

    # Per-language comparison (Recall@10)
    languages = set()
    for r in all_results:
        if r.get("retrieval") and "by_language" in r["retrieval"]:
            languages.update(r["retrieval"]["by_language"].keys())

    if languages:
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.8 / len(languages)
        for i, lang in enumerate(sorted(languages)):
            vals = []
            for r in all_results:
                ret = r.get("retrieval")
                if ret and lang in ret.get("by_language", {}):
                    vals.append(ret["by_language"][lang].get("author_recall_at_10", 0))
                else:
                    vals.append(0)
            offset = (i - len(languages) / 2 + 0.5) * width
            ax.bar([xi + offset for xi in x], vals, width, label=lang.capitalize())

        ax.set_ylabel("Author Recall@10")
        ax.set_title("Embedding Model Comparison — Author Recall@10 by Language")
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.legend()
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(output_dir / "recall_by_language.png", dpi=150)
        log.info("Saved language plot to %s", output_dir / "recall_by_language.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare results across models")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None, help="Save comparison as CSV")
    parser.add_argument("--plot", action="store_true", help="Generate bar chart PNGs")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = config["output_dir"]
    k_values = config["eval"]["k_values"]

    all_results = []
    for model_cfg in config["models"]:
        r = load_model_results(results_dir, model_cfg["name"])
        if r["retrieval"] or r["clustering"]:
            all_results.append(r)
        else:
            log.warning("No results found for %s", model_cfg["name"])

    if not all_results:
        print("No results found. Run eval_retrieval.py and/or eval_clustering.py first.")
        return

    print_comparison_table(all_results, k_values)

    if args.output:
        save_csv(all_results, k_values, args.output)

    if args.plot:
        generate_plots(all_results, k_values, results_dir)


if __name__ == "__main__":
    main()
