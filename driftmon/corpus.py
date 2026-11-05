"""Load the prepared document table together with its embeddings."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from driftmon.config import resolve_path
from driftmon.embeddings import Embedder, document_text


def load_corpus(config: Mapping[str, Any], embedder: Embedder | None = None
                ) -> tuple[pd.DataFrame, np.ndarray]:
    """The cleaned documents (``scripts/prepare_data.py``) and one vector per row.

    Vectors come from the embedding cache, so this is fast after the first
    preparation run. Rows and vectors are aligned by position.
    """
    path = resolve_path(config["paths"]["data_processed"]) / "documents.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/prepare_data.py first")
    docs = pd.read_parquet(path)
    embedder = embedder or Embedder.from_config(config)
    vectors = embedder.embed_documents(
        [document_text(t, x) for t, x in zip(docs["title"], docs["text"])])
    return docs, vectors
