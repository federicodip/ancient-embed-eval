"""Shared helpers: config loading, data I/O, author alias matching, metrics."""

import json
import yaml
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Author alias map
# Keys are canonical English names (lowercased) used in query files.
# Values are substrings that match against the corpus author field.
# Match logic: any(alias in corpus_author.lower() for alias in aliases)
# ---------------------------------------------------------------------------
AUTHOR_ALIASES = {
    "virgil": ["virgil", "vergilius"],
    "horace": ["horace", "horatius"],
    "caesar": ["julius caesar"],
    "cicero": ["cicero", "tullius"],
    "ovid": ["ovid", "ovidius"],
    "seneca": ["seneca"],
    "tacitus": ["tacitus"],
    "juvenal": ["juvenal"],
    "lucretius": ["lucretius"],
    "catullus": ["catullus"],
    "apuleius": ["apuleius"],
    "homer": ["homer"],
    "plato": ["plato"],
    "sophocles": ["sophocles"],
    "euripides": ["euripides"],
    "aristophanes": ["aristophanes"],
    "aeschylus": ["aeschylus"],
    "herodotus": ["herodotus"],
    "thucydides": ["thucydides"],
    "pindar": ["pindar"],
    "hesiod": ["hesiod"],
    "xenophon": ["xenophon"],
    "longus": ["longus"],
    "plutarch": ["plutarch"],
    "petronius": ["petronius"],
    "sallust": ["sallust"],
}


def load_config(path="config.yaml"):
    """Load and return the YAML config as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_model_config(config, model_name):
    """Look up a single model's config block by name. Raises if not found."""
    for m in config["models"]:
        if m["name"] == model_name:
            return m
    raise ValueError(
        f"Model '{model_name}' not in config. "
        f"Available: {[m['name'] for m in config['models']]}"
    )


def load_chunks(path, embed_field="embed_text", limit=None):
    """Load corpus chunks from JSONL.

    Returns a list of dicts. Each dict has at minimum the embed_field,
    'author', 'title', 'work_id', 'language'.
    """
    chunks = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            chunks.append(json.loads(line))
    return chunks


def load_queries(path):
    """Load evaluation queries from JSONL."""
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def authors_match(query_author, corpus_author):
    """Check if a query author matches a corpus author using the alias map.

    query_author: English name from query file (e.g., "Homer", "Virgil")
    corpus_author: scholarly name from corpus (e.g., "P. Vergilius Maro")

    Returns True if they refer to the same person.
    """
    key = query_author.strip().lower()
    corpus_lower = corpus_author.strip().lower()

    # Guard against empty strings (empty author in corpus matches everything)
    if not key or not corpus_lower:
        return False

    # Alias lookup first (more precise than substring matching)
    aliases = AUTHOR_ALIASES.get(key)
    if aliases:
        return any(alias in corpus_lower for alias in aliases)

    # Fallback: direct substring match
    if key in corpus_lower or corpus_lower in key:
        return True

    # Fallback: check if any word in the query author appears in corpus author
    return any(word in corpus_lower for word in key.split() if len(word) > 3)


def works_match(query_work, corpus_title, corpus_work_id=""):
    """Check if a query work matches a corpus work.

    This is harder than author matching because query uses English titles
    ("Iliad") while corpus uses Greek/Latin ("Ἰλιάς") or work IDs.

    Uses a pragmatic approach: check common English-to-original mappings,
    then fall back to substring matching on work_id.
    """
    q = query_work.strip().lower()
    t = corpus_title.strip().lower() if corpus_title else ""
    wid = corpus_work_id.strip().lower() if corpus_work_id else ""

    # Direct match
    if q in t or q in wid:
        return True

    # Common English-to-work_id substring mappings
    WORK_HINTS = {
        "iliad": ["tlg0012.tlg001", "iliad"],
        "odyssey": ["tlg0012.tlg002", "odyss"],
        "republic": ["republic", "respublica", "tlg0059.tlg030"],
        "symposium": ["symposium", "tlg0059.tlg011"],
        "apology": ["apology", "apologia", "tlg0059.tlg002"],
        "aeneid": ["aeneid", "aeneis"],
        "metamorphoses": ["metamorphos"],
        "histories": ["histori"],
        "medea": ["medea"],
        "antigone": ["antigon"],
        "oedipus": ["oedip"],
        "frogs": ["frog", "βάτραχ"],
        "clouds": ["cloud", "νεφέλ"],
        "birds": ["bird", "ὄρνιθ"],
        "odes": ["ode", "carm"],
        "satires": ["satir", "satur"],
        "epistles": ["epistl", "epistul"],
        "gallic war": ["gall"],
        "civil war": ["civil"],
        "annals": ["annal"],
        "germania": ["german"],
        "argonautica": ["argonaut"],
        "theogony": ["theogon"],
        "works and days": ["works and days", "opera et dies", "erga"],
    }

    hints = WORK_HINTS.get(q)
    if hints:
        return any(h in t or h in wid for h in hints)

    return False


def save_embeddings(vectors, metadata, output_dir):
    """Save embedding vectors (.npy) and metadata (.jsonl) to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "vectors.npy", vectors)

    with open(out / "metadata.jsonl", "w", encoding="utf-8") as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


def load_embeddings(model_dir):
    """Load saved embeddings. Returns (vectors: np.ndarray, metadata: list[dict])."""
    model_dir = Path(model_dir)
    vectors = np.load(model_dir / "vectors.npy")

    metadata = []
    with open(model_dir / "metadata.jsonl", encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))

    return vectors, metadata


def compute_mrr(ranks):
    """Mean Reciprocal Rank from a list of ranks (1-indexed). 0 means not found."""
    rrs = []
    for r in ranks:
        rrs.append(1.0 / r if r > 0 else 0.0)
    return np.mean(rrs) if rrs else 0.0


def compute_recall_at_k(hits_at_k, total):
    """Recall@k = fraction of queries with at least one correct hit in top-k."""
    return sum(1 for h in hits_at_k if h) / total if total else 0.0


def compute_ndcg_at_k(ranks, k):
    """NDCG@k for binary relevance (one relevant doc per query).

    For each query, if the correct doc appears at rank r <= k,
    the DCG is 1/log2(r+1). Ideal DCG is always 1/log2(2) = 1.0.
    """
    ndcgs = []
    for r in ranks:
        if 0 < r <= k:
            ndcgs.append(1.0 / np.log2(r + 1))
        else:
            ndcgs.append(0.0)
    return np.mean(ndcgs) if ndcgs else 0.0
