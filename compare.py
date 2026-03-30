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
    """Load retrieval and clustering metrics for a model, including language-filtered results."""
    model_dir = Path(results_dir) / model_name
    result = {"model": model_name}

    # Full corpus results
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

    # Language-filtered results
    for lang in ["latin", "greek"]:
        lang_path = model_dir / lang / "retrieval_metrics.json"
        if lang_path.exists():
            with open(lang_path, encoding="utf-8") as f:
                result[f"retrieval_{lang}"] = json.load(f)
        else:
            result[f"retrieval_{lang}"] = None

    return result


def print_comparison_table(all_results, k_values):
    """Print formatted comparison tables to stdout."""

    # --- Main table: full corpus ---
    header_parts = [f"{'Model':<25}", f"{'Dim':>5}"]
    for k in k_values:
        header_parts.append(f"{'R@'+str(k):>7}")
    header_parts.extend([f"{'MRR':>7}", f"{'ARI':>7}", f"{'NMI':>7}"])
    header = " ".join(header_parts)

    print(f"\n{'='*len(header)}")
    print("FULL CORPUS — Author Recall")
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

    # --- Language-filtered tables ---
    for lang in ["latin", "greek"]:
        key = f"retrieval_{lang}"
        has_data = any(r.get(key) for r in all_results)
        if not has_data:
            continue

        header_parts = [f"{'Model':<25}", f"{'n':>5}"]
        for k in k_values:
            header_parts.append(f"{'R@'+str(k):>7}")
        header_parts.append(f"{'MRR':>7}")
        header = " ".join(header_parts)

        print(f"\n{'='*len(header)}")
        print(f"{lang.upper()}-ONLY CORPUS — Author Recall")
        print(f"{'='*len(header)}")
        print(header)
        print("-" * len(header))

        for r in all_results:
            ret = r.get(key)
            if not ret:
                continue
            parts = [f"{r['model']:<25}"]
            parts.append(f"{ret['num_queries']:>5}")
            for k in k_values:
                val = ret["metrics"].get(f"author_recall_at_{k}", 0)
                parts.append(f"{val:>7.3f}")
            parts.append(f"{ret['metrics'].get('author_mrr', 0):>7.3f}")
            print(" ".join(parts))

    # --- Combined language comparison (compact) ---
    print(f"\n{'='*80}")
    print("LANGUAGE COMPARISON — Author Recall@10 / MRR")
    print(f"{'='*80}")
    print(f"  {'Model':<25} {'Full':>12} {'Latin-only':>12} {'Greek-only':>12}")
    print(f"  {'-'*63}")

    for r in all_results:
        parts = [f"  {r['model']:<25}"]
        # Full
        ret = r.get("retrieval")
        if ret:
            r10 = ret["metrics"].get("author_recall_at_10", 0)
            mrr = ret["metrics"].get("author_mrr", 0)
            parts.append(f"{r10:.3f}/{mrr:.3f}")
        else:
            parts.append(f"{'—':>12}")
        # Latin
        ret_l = r.get("retrieval_latin")
        if ret_l:
            r10 = ret_l["metrics"].get("author_recall_at_10", 0)
            mrr = ret_l["metrics"].get("author_mrr", 0)
            parts.append(f"{r10:.3f}/{mrr:.3f}")
        else:
            parts.append(f"{'—':>12}")
        # Greek
        ret_g = r.get("retrieval_greek")
        if ret_g:
            r10 = ret_g["metrics"].get("author_recall_at_10", 0)
            mrr = ret_g["metrics"].get("author_mrr", 0)
            parts.append(f"{r10:.3f}/{mrr:.3f}")
        else:
            parts.append(f"{'—':>12}")

        print(" ".join(parts))

    print()


def save_csv(all_results, k_values, output_path):
    """Save comparison table as CSV."""
    fieldnames = ["model", "dim"]
    for k in k_values:
        fieldnames.append(f"author_recall_at_{k}")
    fieldnames.extend(["author_mrr", "work_mrr", "author_ari", "author_nmi"])
    for lang in ["latin", "greek"]:
        fieldnames.extend([f"{lang}_recall_at_10", f"{lang}_mrr"])

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

            for lang in ["latin", "greek"]:
                ret_lang = r.get(f"retrieval_{lang}")
                if ret_lang:
                    row[f"{lang}_recall_at_10"] = ret_lang["metrics"].get("author_recall_at_10", "")
                    row[f"{lang}_mrr"] = ret_lang["metrics"].get("author_mrr", "")

            writer.writerow(row)

    log.info("Saved CSV to %s", output_path)


def generate_plots(all_results, k_values, output_dir):
    """Generate comparison bar charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [r["model"] for r in all_results]
    x = range(len(models))

    # --- Plot 1: Recall@k grouped bar chart ---
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
    ax.set_title("Embedding Model Comparison — Author Recall@k (Full Corpus)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_comparison.png", dpi=150)
    log.info("Saved %s", output_dir / "recall_comparison.png")
    plt.close()

    # --- Plot 2: Language comparison (Recall@10, three groups) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    width = 0.25
    colors = {"Full corpus": "#4472C4", "Latin-only": "#ED7D31", "Greek-only": "#70AD47"}

    for i, (label, key) in enumerate([
        ("Full corpus", "retrieval"),
        ("Latin-only", "retrieval_latin"),
        ("Greek-only", "retrieval_greek"),
    ]):
        vals = []
        for r in all_results:
            ret = r.get(key)
            vals.append(ret["metrics"].get("author_recall_at_10", 0) if ret else 0)
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], vals, width, label=label, color=colors[label])

    ax.set_ylabel("Author Recall@10")
    ax.set_title("Author Recall@10 — Full Corpus vs Language-Filtered")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "recall_by_language.png", dpi=150)
    log.info("Saved %s", output_dir / "recall_by_language.png")
    plt.close()

    # --- Plot 3: MRR comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (label, key) in enumerate([
        ("Full corpus", "retrieval"),
        ("Latin-only", "retrieval_latin"),
        ("Greek-only", "retrieval_greek"),
    ]):
        vals = []
        for r in all_results:
            ret = r.get(key)
            vals.append(ret["metrics"].get("author_mrr", 0) if ret else 0)
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], vals, width, label=label, color=colors[label])

    ax.set_ylabel("Author MRR")
    ax.set_title("Author MRR — Full Corpus vs Language-Filtered")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "mrr_by_language.png", dpi=150)
    log.info("Saved %s", output_dir / "mrr_by_language.png")
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
