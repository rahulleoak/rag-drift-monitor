"""Sentence-transformers embeddings with a disk cache.

Embeds documents and queries (queries get the model's instruction prefix from
config). Cached by model name and text hash, so re-runs never re-embed.
Vectors are L2-normalised so inner product equals cosine similarity.
"""
