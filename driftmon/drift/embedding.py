"""Drift technique 1: embedding space (docs/questions.md Q5).

    - MMD with an RBF kernel and a permutation test (distribution-level score
      and p-value)
    - per-document novelty: 1 - highest cosine similarity to the reference
      corpus (gives example documents, and links directly to retrieval)
    - centroid cosine shift (the simple, intuitive number)
    - dispersion: mean pairwise cosine distance, to catch narrowing or
      widening distributions (proposal P3)
"""
