"""Sentence-transformers embeddings with a disk cache.

Embeds documents and queries (queries get the model's instruction prefix from
config). Cached by model name and a hash of the exact text sent to the model,
so re-runs never re-embed. Vectors are L2-normalised so inner product equals
cosine similarity.

Cache layout, one folder per model under ``<cache>/embeddings/``:
    vectors.npy  float32 matrix, one row per cached text
    keys.txt     one SHA-256 key per line, row i of vectors.npy
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

EncodeFn = Callable[[list[str]], np.ndarray]


def document_text(title: str, text: str) -> str:
    """The exact string embedded for a document: headline, then description.

    A full stop is added between them only if the headline does not already
    end with punctuation, so "What If?" does not become "What If?.".
    """
    title, text = title.strip(), text.strip()
    if not text:
        return title
    if not title:
        return text
    sep = " " if title[-1] in ".?!:;" else ". "
    return f"{title}{sep}{text}"


def _key(model_name: str, text: str) -> str:
    return hashlib.sha256(f"{model_name}\n{text}".encode("utf-8")).hexdigest()


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1.0, norms)


class Embedder:
    """Embeds documents and queries with one model, caching every vector on disk.

    Args:
        model_name: a sentence-transformers model id.
        cache_dir: root cache folder; vectors go under ``embeddings/<model>``.
        query_prefix: instruction prepended to queries (empty for models
            that do not use one).
        batch_size: texts per model call.
        encode_fn: optional replacement for the model (used in tests). Takes
            a list of strings and returns one vector per string.
    """

    def __init__(self, model_name: str, cache_dir: str | Path, query_prefix: str = "",
                 batch_size: int = 64, encode_fn: EncodeFn | None = None) -> None:
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._encode_fn = encode_fn
        self.dir = Path(cache_dir) / "embeddings" / re.sub(r"[^A-Za-z0-9._-]+", "__", model_name)
        self._keys: list[str] = []
        self._row: dict[str, int] = {}
        self._vectors = np.zeros((0, 0), dtype=np.float32)
        self.model_calls = 0  # number of texts actually sent to the model
        self._load()

    @classmethod
    def from_config(cls, config: Mapping[str, Any], model_name: str | None = None,
                    encode_fn: EncodeFn | None = None) -> Embedder:
        """Build from the ``embeddings`` section of the config."""
        from driftmon.config import resolve_path
        e = config["embeddings"]
        return cls(model_name or e["model"], resolve_path(config["paths"]["cache"]),
                   query_prefix=e.get("query_prefix") or "", batch_size=e.get("batch_size", 64),
                   encode_fn=encode_fn)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """One normalised vector per document text, from cache where possible."""
        return self._embed(list(texts))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """One normalised vector per query; the query prefix is added first."""
        return self._embed([self.query_prefix + t for t in texts])

    def _embed(self, texts: list[str]) -> np.ndarray:
        keys = [_key(self.model_name, t) for t in texts]
        missing: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in self._row and k not in missing:
                missing[k] = t
        if missing:
            new = _normalise(self._encode(list(missing.values())))
            self._add(list(missing), new)
            self._save()
        if not texts:
            return np.zeros((0, self._vectors.shape[1]), dtype=np.float32)
        return self._vectors[[self._row[k] for k in keys]]

    def _encode(self, texts: list[str]) -> np.ndarray:
        self.model_calls += len(texts)
        if self._encode_fn is None:
            self._encode_fn = self._load_model()
        out = [self._encode_fn(texts[i:i + self.batch_size])
               for i in range(0, len(texts), self.batch_size)]
        return np.vstack(out)

    def _load_model(self) -> EncodeFn:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(self.model_name, device=device)

        def encode(batch: list[str]) -> np.ndarray:
            return model.encode(batch, batch_size=len(batch), normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
        return encode

    def _add(self, keys: list[str], vectors: np.ndarray) -> None:
        start = len(self._keys)
        self._vectors = vectors if start == 0 else np.vstack([self._vectors, vectors])
        self._keys.extend(keys)
        self._row.update({k: start + i for i, k in enumerate(keys)})

    def _load(self) -> None:
        keys_file, vec_file = self.dir / "keys.txt", self.dir / "vectors.npy"
        if not (keys_file.exists() and vec_file.exists()):
            return
        keys = keys_file.read_text(encoding="utf-8").split()
        vectors = np.load(vec_file)
        if len(keys) != len(vectors):  # a half-written cache: start again rather than misalign
            return
        self._add(keys, vectors.astype(np.float32))

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        # write to temporary files first so an interrupted run cannot leave
        # keys and vectors out of step
        tmp_vec, tmp_keys = self.dir / "vectors.tmp.npy", self.dir / "keys.tmp.txt"
        np.save(tmp_vec, self._vectors)
        tmp_keys.write_text("\n".join(self._keys), encoding="utf-8")
        tmp_vec.replace(self.dir / "vectors.npy")
        tmp_keys.replace(self.dir / "keys.txt")
