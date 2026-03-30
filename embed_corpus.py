#!/usr/bin/env python3
"""Embed corpus chunks with a specified model and save vectors to disk.

Usage:
    python embed_corpus.py --model bge-m3               # one model
    python embed_corpus.py --model all                   # all models in config
    python embed_corpus.py --model bge-m3 --sample 5000  # quick test on subset
    python embed_corpus.py --model bge-m3 --resume       # resume interrupted run
"""

import argparse
import json
import time
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import load_config, get_model_config, load_chunks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def load_st_model(model_cfg):
    """Load a SentenceTransformer model with the right kwargs for each model type.

    Some newer models (Qwen3, Nemotron) need trust_remote_code and bfloat16.
    This builds the correct kwargs from the config block.
    """
    hf_id = model_cfg["hf_id"]
    model_kwargs = {}
    tokenizer_kwargs = {}

    if model_cfg.get("torch_dtype") == "bfloat16":
        model_kwargs["torch_dtype"] = "bfloat16"

    if model_cfg.get("trust_remote_code"):
        model_kwargs["trust_remote_code"] = True
        tokenizer_kwargs["trust_remote_code"] = True

    # Qwen3 and Nemotron need left-padding for decoder-based architectures
    encode_mode = model_cfg.get("encode_mode", "standard")
    if encode_mode in ("prompt_name", "asymmetric"):
        tokenizer_kwargs["padding_side"] = "left"

    st_kwargs = {"model_kwargs": model_kwargs} if model_kwargs else {}
    if tokenizer_kwargs:
        st_kwargs["tokenizer_kwargs"] = tokenizer_kwargs
    if model_cfg.get("trust_remote_code"):
        st_kwargs["trust_remote_code"] = True

    log.info("Loading model %s (%s) with kwargs: %s", model_cfg["name"], hf_id, st_kwargs)
    model = SentenceTransformer(hf_id, **st_kwargs)
    log.info("Model loaded. Embedding dim: %d", model.get_sentence_embedding_dimension())
    return model


def encode_passages(model, texts, model_cfg):
    """Encode passage texts using the correct mode for this model.

    - standard:    model.encode(texts) with optional passage_prefix prepended
    - prompt_name: model.encode(texts) with no prompt (passages don't get one)
    - asymmetric:  model.encode_document(texts) — dedicated method
    """
    batch_size = model_cfg.get("batch_size", 64)
    normalize = model_cfg.get("normalize", True)
    encode_mode = model_cfg.get("encode_mode", "standard")

    if encode_mode == "asymmetric":
        return model.encode_document(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )
    else:
        # "standard" and "prompt_name" both use plain encode() for passages
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=normalize,
        )


def embed_model(model_cfg, config, args):
    """Embed the corpus with a single model."""
    name = model_cfg["name"]
    hf_id = model_cfg["hf_id"]
    batch_size = model_cfg.get("batch_size", 64)
    normalize = model_cfg.get("normalize", True)
    encode_mode = model_cfg.get("encode_mode", "standard")
    passage_prefix = model_cfg.get("passage_prefix", "")

    embeddings_dir = Path(config["embeddings_dir"]) / name
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    vectors_path = embeddings_dir / "vectors.npy"
    metadata_path = embeddings_dir / "metadata.jsonl"
    info_path = embeddings_dir / "embed_info.json"

    # Load corpus
    limit = args.sample if args.sample else None
    log.info("Loading corpus from %s (limit=%s)", config["corpus"], limit)
    chunks = load_chunks(config["corpus"], config["embed_field"], limit=limit)
    log.info("Loaded %d chunks", len(chunks))

    # Resume support: skip already-embedded chunks
    start_idx = 0
    if args.resume and vectors_path.exists():
        existing = np.load(vectors_path)
        start_idx = existing.shape[0]
        log.info("Resuming from chunk %d (found %d existing vectors)", start_idx, start_idx)
        if start_idx >= len(chunks):
            log.info("All chunks already embedded. Skipping %s.", name)
            return

    # Load model
    model = load_st_model(model_cfg)

    # Prepare texts and metadata
    embed_field = config["embed_field"]
    texts = []
    metadata = []
    for chunk in chunks[start_idx:]:
        text = chunk.get(embed_field, "")
        # Only prepend passage_prefix for standard mode (asymmetric/prompt_name
        # handle prefixing internally via their encode methods)
        if encode_mode == "standard" and passage_prefix:
            text = passage_prefix + text
        texts.append(text)
        metadata.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "work_id": chunk.get("work_id", ""),
            "author": chunk.get("author", ""),
            "title": chunk.get("title", ""),
            "language": chunk.get("language", ""),
        })

    # Embed
    log.info("Embedding %d texts (mode=%s, batch=%d, normalize=%s)",
             len(texts), encode_mode, batch_size, normalize)
    t0 = time.time()

    vectors = encode_passages(model, texts, model_cfg)

    elapsed = time.time() - t0
    log.info("Embedding done in %.1fs (%.0f chunks/sec)", elapsed, len(texts) / elapsed)

    # If resuming, concatenate with existing vectors
    if start_idx > 0:
        existing = np.load(vectors_path)
        vectors = np.vstack([existing, vectors])

        existing_meta = []
        with open(metadata_path, encoding="utf-8") as f:
            for line in f:
                existing_meta.append(json.loads(line))
        metadata = existing_meta + metadata

    # Save
    vectors = vectors.astype(np.float32)
    np.save(vectors_path, vectors)
    log.info("Saved vectors: %s %s", vectors_path, vectors.shape)

    with open(metadata_path, "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    log.info("Saved metadata: %s (%d records)", metadata_path, len(metadata))

    # Save run info
    info = {
        "model": name,
        "hf_id": hf_id,
        "encode_mode": encode_mode,
        "corpus": config["corpus"],
        "num_chunks": len(metadata),
        "embedding_dim": int(vectors.shape[1]),
        "normalize": normalize,
        "batch_size": batch_size,
        "passage_prefix": passage_prefix,
        "elapsed_seconds": round(elapsed, 1),
        "chunks_per_second": round(len(texts) / elapsed, 1),
        "sample": args.sample,
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Embed corpus chunks with embedding models")
    parser.add_argument("--model", required=True, help="Model name from config.yaml, or 'all'")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--sample", type=int, default=None, help="Only embed first N chunks (for testing)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing partial embeddings")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model == "all":
        for model_cfg in config["models"]:
            embed_model(model_cfg, config, args)
    else:
        model_cfg = get_model_config(config, args.model)
        embed_model(model_cfg, config, args)


if __name__ == "__main__":
    main()
