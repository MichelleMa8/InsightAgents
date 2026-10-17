"""
Qwen3-Embedding-0.6B query embedder (CPU). Queries are encoded with last-token
pooling over name and description joined by a newline, matching the precomputed
corpus shards; a mismatched pooling still gives plausible cosines but ranks in a
different space.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# Qwen3-Embedding is trained for *asymmetric* retrieval: the query carries a task
# instruction, documents are embedded bare. The probe above confirms the corpus
# side is bare, so this prefix belongs on the query side only.
#
# It is not cosmetic. Matching the pooling makes the space correct; the
# instruction is what makes it discriminative. Without it, query cosines are
# compressed: on-topic papers rank no better and a large share of *random*
# contributions clears CG_SIMILARITY_THRESHOLD. That matters beyond ranking:
# the threshold gates graph growth in expertise/contribution_graph.py against
# this same vector, so an uninstructed query would admit arbitrary papers.
QUERY_INSTRUCTION = "Given a research topic, retrieve relevant scientific contributions"

# The corpus shards were embedded at max_length=1024 (data/.../embeddings/manifest.json).
# Queries are far shorter than this, but the cap is matched rather than guessed so a
# long expanded query cannot be truncated differently from the corpus side.
MAX_LENGTH = 1024

_tokenizer = None
_model     = None


def _load_model() -> None:
    global _tokenizer, _model
    if _model is not None:
        return
    logger.info("Loading %s (CPU) …", MODEL_NAME)
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    _model     = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    _model.eval()
    logger.info("Embedding model loaded.")


def embed_queries(texts: List[str], instruction: str = QUERY_INSTRUCTION) -> np.ndarray:
    """
    Embed a list of query strings into the corpus's embedding space.

    Returns float32 numpy array of shape (len(texts), 1024), L2-normalised, so a
    dot product against a corpus vector is a cosine.

    instruction: task instruction prefixed to every query (Qwen3-Embedding's
                 asymmetric convention). Defaults to QUERY_INSTRUCTION; every
                 caller in this repo embeds queries, never documents, so the
                 default is what you want. Pass "" to embed text the way the
                 *corpus* side was embedded — needed only to verify that this
                 function still lands in the shards' space (re-encode a
                 contribution and compare against its stored shard vector).
    """
    _load_model()

    if instruction:
        texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

    # Right padding, so the last real token sits at index (mask.sum() - 1) below.
    _tokenizer.padding_side = "right"
    inputs = _tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    # Last-token pooling, matching how the corpus shards were built (see module
    # docstring). Indexed off the attention mask instead of taking h[:, -1], so a
    # batch cannot silently pool a pad position if padding_side is ever changed.
    hidden        = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"]
    last_idx      = attention_mask.sum(dim=1) - 1
    embeddings    = hidden[torch.arange(hidden.shape[0]), last_idx].numpy().astype(np.float32)

    # L2 normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-9)
    return embeddings / norms
